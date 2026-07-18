import argparse
import csv
from pathlib import Path

import torch
import yaml
from PIL import Image

from base_tool.archs import build_network
from base_tool.data.preprocessing import build_image_transform, resolve_preprocessing


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


def _load_class_names_from_csv(class_index_path):
    with class_index_path.open('r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        rows = sorted(reader, key=lambda row: int(row['class_index']))
        return [row['class_name'] for row in rows]


def _class_names_from_checkpoint(checkpoint):
    idx_to_class = checkpoint.get('idx_to_class') or checkpoint.get('dataset_metadata', {}).get('idx_to_class')
    if not idx_to_class:
        return None
    return [idx_to_class[key] for key in sorted(idx_to_class, key=lambda value: int(value))]


def main():
    parser = argparse.ArgumentParser(description='Classify an image using a trained checkpoint.')
    parser.add_argument('-opt', required=True, help='Path to the training YAML used to build the model.')
    parser.add_argument('--checkpoint', required=True, help='Path to a .pth checkpoint.')
    parser.add_argument('--image', required=True, help='Path to the image to classify.')
    parser.add_argument('--class-index', default=None, help='Optional legacy path to class_index.csv.')
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[1]
    with open(args.opt, mode='r', encoding='utf-8') as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_network(opt['network_g']).to(device)
    checkpoint = torch.load(_resolve_path(root_path, args.checkpoint), map_location=device)
    state_dict = checkpoint.get('network', checkpoint)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    class_names = _class_names_from_checkpoint(checkpoint)
    if class_names is None:
        dataset_root = _resolve_path(root_path, opt['datasets']['train']['root']).parent
        class_index_path = _resolve_path(root_path, args.class_index) if args.class_index else dataset_root / 'class_index.csv'
        class_names = _load_class_names_from_csv(class_index_path)

    preprocessing = checkpoint.get('preprocessing') or resolve_preprocessing(opt)
    transform = build_image_transform(preprocessing, augment=False)
    with Image.open(_resolve_path(root_path, args.image)) as image:
        tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    print(f'class={class_names[predicted_index.item()]} confidence={confidence.item():.4f}')


if __name__ == '__main__':
    main()
