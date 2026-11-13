import torch
import torch.nn as nn
import timm

from base_tool.archs.coordinate_attention import CoordinateAttention
from base_tool.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class TimmCoordinateAttentionClassifier(nn.Module):
    """
    Classificador mobile Timm (MobileNetV4, RepViT, etc.) com bloco Coordinate Attention (CA)
    injetado nas features espaciais antes do pooling e head de classificação.
    Fatora o pooling 2D em 1D horizontal e vertical, preservando pistas posicionais
    da fauna amazônica com custo computacional desprezível.
    """

    def __init__(
        self,
        architecture: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        num_classes: int = 16,
        pretrained: bool = True,
        dropout: float = 0.20,
        reduction: int = 16,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.architecture = architecture
        self.num_classes = int(num_classes)

        # 1. Instanciar modelo Timm
        self.model = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=self.num_classes,
            drop_rate=float(dropout),
        )

        # 2. Injetar Coordinate Attention nas features
        num_features = self.model.num_features
        self.ca = CoordinateAttention(
            in_channels=num_features,
            out_channels=num_features,
            reduction=reduction,
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # Manter o bloco CA e a cabeça classificadora treináveis
        for param in self.ca.parameters():
            param.requires_grad = True
        head = self.model.get_classifier()
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True

    def get_classifier(self):
        return self.model.get_classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extrai mapa de features [B, C, H, W]
        feat = self.model.forward_features(x)
        # Aplica Coordinate Attention espacial direcional
        feat_ca = self.ca(feat)
        # Passa pelo pooling global e camada linear de saída
        return self.model.forward_head(feat_ca)
