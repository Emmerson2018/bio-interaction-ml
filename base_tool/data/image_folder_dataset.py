from pathlib import Path

from PIL import Image
from torchvision import transforms

from base_tool.data.base_dataset import BaseDataset
from base_tool.utils.registry import DATASET_REGISTRY


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@DATASET_REGISTRY.register()
class ImageFolderClassificationDataset(BaseDataset):
    def __init__(self, opt):
        super(ImageFolderClassificationDataset, self).__init__(opt)
        self.root = Path(opt['root'])
        self.image_size = int(opt.get('image_size', 224))
        self.augment = bool(opt.get('augment', False))
        self.classes = sorted([path.name for path in self.root.iterdir() if path.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self._load_samples()
        self.transform = self._build_transform()

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

    def _build_transform(self):
        use_grayscale = bool(self.opt.get('grayscale', False))
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if use_grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=10),
            ])
        transform_list.extend([
            transforms.ToTensor(),
        ])
        if self.augment:
            transform_list.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)))

        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
            
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            image = self.transform(image)
        return {'x': image, 'y': target, 'path': str(image_path)}
