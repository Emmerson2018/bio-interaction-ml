import torch.nn as nn
from base_tool.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class SimpleMLP(nn.Module):
    def __init__(self, in_channels=10, out_channels=1, hidden_channels=32):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)
