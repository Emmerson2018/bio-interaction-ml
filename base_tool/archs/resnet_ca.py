import torch
import torch.nn as nn
from torchvision import models

from base_tool.archs.coordinate_attention import CoordinateAttention
from base_tool.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ResNet18CA(nn.Module):
    """
    SOTA 2: ResNet18 com blocos Coordinate Attention (CA) embutidos em layer3 e layer4.
    Preserva a inicialização pré-treinada do ImageNet, adicionando atenção espacial
    direcional com sobrecarga de parâmetros < 1.5% e < 0.8 ms de latência mobile.
    """

    def __init__(
        self,
        num_classes: int = 15,
        weights: str = 'DEFAULT',
        dropout: float = 0.15,
        freeze_backbone: bool = False,
    ):
        super(ResNet18CA, self).__init__()
        weights_enum = models.ResNet18_Weights.DEFAULT if weights == 'DEFAULT' else None
        self.backbone = models.resnet18(weights=weights_enum)

        # Injetar blocos de Coordinate Attention nas saídas de layer3 (256 canais) e layer4 (512 canais)
        self.backbone.layer3 = nn.Sequential(
            self.backbone.layer3,
            CoordinateAttention(in_channels=256, out_channels=256, reduction=16),
        )
        self.backbone.layer4 = nn.Sequential(
            self.backbone.layer4,
            CoordinateAttention(in_channels=512, out_channels=512, reduction=16),
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
