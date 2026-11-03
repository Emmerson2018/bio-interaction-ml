import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design (Hou et al., CVPR 2021).
    Fatora o pooling espacial 2D em dois poolings unidimensionais direcionais (X e Y),
    preservando informações posicionais precisas com sobrecarga desprezível (< 1 ms no mobile).
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HardSwish()

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # 1. Poolings direcionais independentes ao longo de altura e largura
        x_h = self.pool_h(x)                          # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # [N, C, W, 1]

        # 2. Concatenação ao longo da dimensão espacial
        y = torch.cat([x_h, x_w], dim=2)              # [N, C, H + W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 3. Separação de volta nas componentes vertical e horizontal
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 4. Projeções convolucionais com Sigmoid
        a_h = torch.sigmoid(self.conv_h(x_h))         # [N, C, H, 1]
        a_w = torch.sigmoid(self.conv_w(x_w))         # [N, C, 1, W]

        # 5. Ponderação espacial atenta
        out = identity * a_h * a_w
        return out
