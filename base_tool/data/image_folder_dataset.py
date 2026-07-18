from pathlib import Path

from PIL import Image

from base_tool.data.base_dataset import BaseDataset
from base_tool.data.preprocessing import build_image_transform, resolve_preprocessing
from base_tool.utils.registry import DATASET_REGISTRY


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@DATASET_REGISTRY.register()
class ImageFolderClassificationDataset(BaseDataset):
    def __init__(self, opt):
        super(ImageFolderClassificationDataset, self).__init__(opt)
        self.root = Path(opt['root'])
        self.augment = bool(opt.get('augment', False))
        preprocessing_opt = {'network_g': opt.get('network_g', {}), 'dataset': opt}
        self.preprocessing = resolve_preprocessing(preprocessing_opt)
        self.classes = sorted([path.name for path in self.root.iterdir() if path.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.samples = self._load_samples()
        self.transform = build_image_transform(self.preprocessing, augment=self.augment)

        if not self.samples:
            raise FileNotFoundError(f'No images found in {self.root}')

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for path in sorted(class_dir.rglob('*')):
                if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((path, self.class_to_idx[class_name]))
        return samples

    def get_metadata(self):
        return {
            'classes': list(self.classes),
            'class_to_idx': dict(self.class_to_idx),
            'idx_to_class': {str(idx): class_name for idx, class_name in self.idx_to_class.items()},
            'preprocessing': dict(self.preprocessing),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            image = self.transform(image)
        return {'x': image, 'y': target, 'path': str(image_path)}
