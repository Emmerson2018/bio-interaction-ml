import argparse
import csv
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from base_tool.archs import build_network


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


def _load_class_names(class_index_path):
    with class_index_path.open('r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        rows = sorted(reader, key=lambda row: int(row['class_index']))
        return [row['class_name'] for row in rows]


def _build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description='Classify an image using a trained checkpoint.')
    parser.add_argument('-opt', required=True, help='Path to the training YAML used to build the model.')
    parser.add_argument('--checkpoint', required=True, help='Path to a .pth checkpoint.')
    parser.add_argument('--image', required=True, help='Path to the image to classify.')
    parser.add_argument('--class-index', default=None, help='Path to class_index.csv.')
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[1]
    with open(args.opt, mode='r', encoding='utf-8') as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    dataset_root = _resolve_path(root_path, opt['datasets']['train']['root']).parent
    class_index_path = _resolve_path(root_path, args.class_index) if args.class_index else dataset_root / 'class_index.csv'
    class_names = _load_class_names(class_index_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_network(opt['network_g']).to(device)
    checkpoint = torch.load(_resolve_path(root_path, args.checkpoint), map_location=device)
    state_dict = checkpoint.get('network', checkpoint)
    net.load_state_dict(state_dict)
    net.eval()

    image_size = int(opt['datasets']['val'].get('image_size', opt['datasets']['train'].get('image_size', 224)))
    transform = _build_transform(image_size)
    with Image.open(_resolve_path(root_path, args.image)) as image:
        tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    print(f'class={class_names[predicted_index.item()]} confidence={confidence.item():.4f}')


if __name__ == '__main__':
    main()
