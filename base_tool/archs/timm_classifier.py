import torch.nn as nn
import timm

from base_tool.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class TimmClassifier(nn.Module):
    """
    Classificador mobile integrado ao catálogo oficial do timm.
    Suporta modelos SOTA móveis:
    - mobilenetv4_conv_small.e2400_r224_in1k (3.8M params, blocos UIB)
    - repvit_m0_9 (4.7M params, reparametrizável)
    - convnext_tiny
    - efficientnet_b0
    """

    def __init__(
        self,
        architecture: str = 'mobilenetv4_conv_small.e2400_r224_in1k',
        num_classes: int = 16,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.architecture = architecture
        self.num_classes = int(num_classes)
        self.model = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=self.num_classes,
            drop_rate=float(dropout),
        )
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        head = self.model.get_classifier()
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True

    def get_classifier(self):
        return self.model.get_classifier()

    def forward(self, x):
        return self.model(x)
