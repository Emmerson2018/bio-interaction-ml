import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from base_tool.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class TwoStageCropClassifier(nn.Module):
    """
    SOTA 1: Pipeline em dois estágios.
    1. Detector ultraleve (YOLOv8n) para identificar a Bounding Box da fauna na imagem.
    2. Recorte com Margem Dinâmica (+15% padding) e redimensionamento para 224x224.
    3. Classificador ResNet18 operando exclusivamente na Região de Interesse (ROI).
    """

    def __init__(
        self,
        classifier_arch: str = 'resnet18',
        num_classes: int = 15,
        weights: str = 'DEFAULT',
        dropout: float = 0.15,
        crop_margin: float = 0.15,
        conf_thresh: float = 0.25,
        detector_weights: str = 'yolov8n.pt',
    ):
        super(TwoStageCropClassifier, self).__init__()
        self.crop_margin = float(crop_margin)
        self.conf_thresh = float(conf_thresh)

        # 1. Carregar classificador base (ResNet18)
        self.classifier = models.get_model(classifier_arch, weights=weights)
        in_features = self.classifier.fc.in_features
        self.classifier.fc = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, num_classes),
        )

        # 2. Inicializar detector YOLO em container seguro fora de _modules do PyTorch
        # (YOLO herda de nn.Module mas NÃO deve ser registrado como submódulo para evitar
        # poluição de state_dict, vazamento de memória e interferência no Model EMA)
        self._detector_holder = [None]
        self.detector_weights = detector_weights

    @property
    def detector(self):
        return self._detector_holder[0]

    def train(self, mode: bool = True):
        self.training = mode
        self.classifier.train(mode)
        det = self.detector
        if det and hasattr(det, 'model') and isinstance(det.model, nn.Module):
            det.model.eval()
        return self

    def _ensure_detector_loaded(self, device):
        if self._detector_holder[0] is None:
            try:
                from ultralytics import YOLO
                det = YOLO(self.detector_weights)
                if hasattr(det, 'model') and isinstance(det.model, nn.Module):
                    for param in det.model.parameters():
                        param.requires_grad = False
                    det.model.eval()
                self._detector_holder[0] = det
            except Exception as e:
                print(f"[TwoStageCropClassifier] Aviso ao carregar YOLO: {e}. Usando fallback central crop.")
                self._detector_holder[0] = False

    def extract_crops(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recebe lote [B, C, H, W] em float [0, 1] e retorna as ROIs recortadas [B, C, 224, 224].
        """
        B, C, H, W = x.shape
        self._ensure_detector_loaded(x.device)

        crops = []
        for i in range(B):
            img = x[i]
            box = None

            if self.detector:
                try:
                    # Converter tensor PyTorch para formato aceito pelo YOLO (H, W, C uint8 na CPU)
                    img_np = (torch.clamp(img, 0.0, 1.0) * 255.0).permute(1, 2, 0).byte().cpu().numpy()
                    results = self.detector.predict(
                        img_np,
                        conf=self.conf_thresh,
                        verbose=False,
                        imgsz=256,
                    )
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        # Selecionar a caixa de maior confiança
                        boxes = results[0].boxes
                        best_idx = torch.argmax(boxes.conf).item()
                        xyxy = boxes.xyxy[best_idx].cpu().numpy()
                        box = [float(coord) for coord in xyxy]
                except Exception:
                    box = None

            if box is not None:
                bx1, by1, bx2, by2 = box
                bw = bx2 - bx1
                bh = by2 - by1
                pad_w = bw * self.crop_margin
                pad_h = bh * self.crop_margin

                x1 = max(0, int(bx1 - pad_w))
                y1 = max(0, int(by1 - pad_h))
                x2 = min(W, int(bx2 + pad_w))
                y2 = min(H, int(by2 + pad_h))

                # Garantir que a área é válida
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    crop = img[:, y1:y2, x1:x2]
                else:
                    crop = img[:, int(0.1 * H) : int(0.9 * H), int(0.1 * W) : int(0.9 * W)]
            else:
                # Fallback: Central crop de 80% da imagem
                crop = img[:, int(0.1 * H) : int(0.9 * H), int(0.1 * W) : int(0.9 * W)]

            # Redimensionar para 224x224
            crop_resized = F.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
            )
            crops.append(crop_resized)

        return torch.cat(crops, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extrai os crops adaptativos
        crops = self.extract_crops(x)
        # Classifica a ROI
        return self.classifier(crops)
