from torchvision import transforms

from base_tool.archs.image_classifier import resolve_torchvision_preprocessing


def resolve_preprocessing(opt):
    if opt is None:
        opt = {}

    network_opt = opt.get('network_g', opt)
    backbone = network_opt.get('backbone') or network_opt.get('architecture') or 'resnet18'
    weights = network_opt.get('weights')
    preprocessing = dict(network_opt.get('preprocessing') or {})

    dataset_opt = opt.get('dataset') or {}
    image_size = dataset_opt.get('input_size') or dataset_opt.get('image_size')
    if image_size is not None and 'input_size' not in preprocessing:
        preprocessing['input_size'] = int(image_size)
        preprocessing['resize_size'] = int(image_size)
        preprocessing['crop_size'] = int(image_size)

    return resolve_torchvision_preprocessing(backbone, weights, preprocessing)


def build_image_transform(preprocessing, augment=False):
    input_size = int(preprocessing.get('input_size', 224))
    interpolation = transforms.InterpolationMode.BILINEAR

    transform_list = [
        transforms.Resize((input_size, input_size), interpolation=interpolation),
    ]
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=preprocessing['mean'], std=preprocessing['std']),
    ])
    return transforms.Compose(transform_list)
