import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import MobileNet_V3_Small_Weights
from typing import Dict, List, Optional, Tuple


class ExemplarVerifier:
    """
    Verificador de Identidade Visual e Detector de Oclusão (Exemplar Matching).
    
    Utiliza o primeiro recorte confirmado como 'Gabarito de Ouro' e avalia
    cada candidato subsequente através de:
    1. Invariantes Geométricos (área normalizada e penetração de borda).
    2. Similaridade de Cosseno de Embeddings Profundos (MobileNetV3-Small).
    3. Consistência Cromática (Histograma HSV em Matiz e Saturação).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sim_threshold: float = 0.50,
        min_norm_area: float = 0.012,
        max_norm_area: float = 0.80,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sim_threshold = float(sim_threshold)
        self.min_norm_area = float(min_norm_area)
        self.max_norm_area = float(max_norm_area)

        # Extrator leve MobileNetV3-Small (9.8MB, ~3.7ms na GPU)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = tv_models.mobilenet_v3_small(weights=weights).to(self.device).eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

        self.exemplar_feat: Optional[torch.Tensor] = None
        self.exemplar_hist: Optional[np.ndarray] = None
        self.exemplar_crop_bgr: Optional[np.ndarray] = None

    def _extract_deep_embedding(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Extrai vetor L2-normalizado de 576 dimensões a partir de um recorte BGR 224x224."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (224, 224):
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        t = (t - self.mean) / self.std
        t = t.unsqueeze(0)

        with torch.no_grad():
            feat = self.model.features(t)
            feat = self.model.avgpool(feat)
            feat = torch.flatten(feat, 1)
            feat = F.normalize(feat, p=2, dim=1)

        return feat

    def _extract_color_histogram(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrai histograma 2D normalizado de Matiz (Hue) e Saturação (Sat) em HSV."""
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask_uint8 = None
        if mask is not None:
            mask_uint8 = (mask > 0).astype(np.uint8)
            if mask_uint8.shape[:2] != crop_bgr.shape[:2]:
                mask_uint8 = cv2.resize(mask_uint8, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        hist = cv2.calcHist([hsv], [0, 1], mask_uint8, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def set_exemplar(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None):
        """Registra o Gabarito de Ouro do primeiro frame para o vídeo atual."""
        self.exemplar_crop_bgr = crop_bgr.copy()
        self.exemplar_feat = self._extract_deep_embedding(crop_bgr)
        self.exemplar_hist = self._extract_color_histogram(crop_bgr, mask)

    def has_exemplar(self) -> bool:
        return self.exemplar_feat is not None

    def reset(self):
        self.exemplar_feat = None
        self.exemplar_hist = None
        self.exemplar_crop_bgr = None

    def verify_candidate(
        self,
        candidate_crop_bgr: np.ndarray,
        candidate_box_xyxy: List[float],
        frame_shape: Tuple[int, int],
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Avalia se a região candidata representa o mesmo animal do gabarito.
        Retorna dicionário com status de validade, escore de similaridade e motivo.
        """
        if not self.has_exemplar():
            # Sem gabarito registrado, aceita como neutro
            return {
                'is_valid': True,
                'similarity': 1.0,
                'cosine_sim': 1.0,
                'color_sim': 1.0,
                'reason': 'NO_EXEMPLAR_SET',
            }

        orig_h, orig_w = frame_shape[:2]
        bx1, by1, bx2, by2 = candidate_box_xyxy
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        norm_area = (bw * bh) / float(orig_w * orig_h)

        # 1. Invariantes Geométricos
        if norm_area < self.min_norm_area:
            return {
                'is_valid': False,
                'similarity': 0.0,
                'cosine_sim': 0.0,
                'color_sim': 0.0,
                'reason': f'TINY_AREA ({norm_area*100:.2f}% < {self.min_norm_area*100:.1f}%)',
            }

        if norm_area > self.max_norm_area:
            return {
                'is_valid': False,
                'similarity': 0.0,
                'cosine_sim': 0.0,
                'color_sim': 0.0,
                'reason': f'EXPLODED_AREA ({norm_area*100:.2f}% > {self.max_norm_area*100:.1f}%)',
            }

        # Checagem de colapso colado na borda (se a caixa estiver encostando na margem e for pequena)
        touches_left = (bx1 <= 4)
        touches_top = (by1 <= 4)
        touches_right = (bx2 >= orig_w - 4)
        touches_bottom = (by2 >= orig_h - 4)
        border_count = sum([touches_left, touches_top, touches_right, touches_bottom])

        if border_count >= 1 and norm_area < 0.05:
            return {
                'is_valid': False,
                'similarity': 0.0,
                'cosine_sim': 0.0,
                'color_sim': 0.0,
                'reason': 'BORDER_STUCK_DRIFT',
            }

        # 2. Similaridade de Embedding Profundo (MobileNetV3)
        candidate_feat = self._extract_deep_embedding(candidate_crop_bgr)
        cosine_sim = (candidate_feat @ self.exemplar_feat.T).item()

        # 3. Similaridade de Cor (Histograma HSV)
        candidate_hist = self._extract_color_histogram(candidate_crop_bgr, mask)
        color_sim_raw = cv2.compareHist(self.exemplar_hist, candidate_hist, cv2.HISTCMP_CORREL)
        color_sim = float(max(0.0, min(1.0, color_sim_raw)))

        # 4. Métrica Combinada Híbrida: 85% semântica visual profunda + 15% consistência cromática
        combined_sim = float(0.85 * cosine_sim + 0.15 * color_sim)

        is_valid = (combined_sim >= self.sim_threshold) and (cosine_sim >= (self.sim_threshold - 0.05))

        reason = 'VALID_MATCH' if is_valid else f'LOW_SIMILARITY ({combined_sim:.2f} < {self.sim_threshold:.2f})'

        return {
            'is_valid': is_valid,
            'similarity': round(combined_sim, 3),
            'cosine_sim': round(cosine_sim, 3),
            'color_sim': round(color_sim, 3),
            'reason': reason,
        }
