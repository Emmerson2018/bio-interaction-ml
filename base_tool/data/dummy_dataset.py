import torch
from base_tool.data.base_dataset import BaseDataset
from base_tool.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class DummyDataset(BaseDataset):
    def __init__(self, opt):
        super(DummyDataset, self).__init__(opt)
        self.num_samples = opt.get('num_samples', 100)
        self.in_channels = opt.get('in_channels', 10)
        self.out_channels = opt.get('out_channels', 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = torch.randn(self.in_channels)
        y = torch.randn(self.out_channels)
        return {'x': x, 'y': y}
