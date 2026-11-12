from torch.utils.data import DataLoader
from base_tool.utils.registry import DATASET_REGISTRY
from base_tool.utils.misc import import_modules_from_directory
import os

# Importa automaticamente todos os datasets
data_folder = os.path.dirname(os.path.abspath(__file__))
import_modules_from_directory(data_folder, 'base_tool.data')

def build_dataset(opt):
    """Constroi um dataset."""
    dataset_type = opt['type']
    dataset = DATASET_REGISTRY.get(dataset_type)(opt)
    return dataset

def build_dataloader(dataset, opt, phase):
    """Constroi um dataloader."""
    if phase == 'train':
        batch_size = opt['datasets']['train'].get('batch_size_per_gpu', 1)
        requested_workers = opt['datasets']['train'].get('num_worker_per_gpu', 0)
        max_cpus = os.cpu_count() or 1
        num_worker = min(requested_workers, max_cpus)
        shuffle = True
    else:
        batch_size = 1
        num_worker = 0
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        pin_memory=True
    )
