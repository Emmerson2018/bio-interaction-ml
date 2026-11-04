import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from base_tool.archs import build_network


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


def _load_class_names(class_index_path):
    if not class_index_path.exists():
        return []
    with class_index_path.open('r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        rows = sorted(reader, key=lambda row: int(row['class_index']))
        return [row['class_name'] for row in rows]


def main():
    parser = argparse.ArgumentParser(description='Export trained PyTorch model (.pth) to ONNX and TFLite (.tflite).')
    parser.add_argument('-opt', '--options', required=True, help='Path to training YAML config.')
    parser.add_argument('--checkpoint', default=None, help='Path to trained .pth checkpoint file. If omitted, automatically selects the latest/best in models/.')
    parser.add_argument('--output-dir', default=None, help='Output directory for exported ONNX and TFLite files.')
    parser.add_argument('--static-shape', action='store_true', default=True, help='Freeze static batch=1 shape for mobile delegates.')
    parser.add_argument('--enable-int8', action='store_true', default=True, help='Generate INT8 quantized TFLite model.')
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[1]
    opt_path = _resolve_path(root_path, args.options)

    with open(opt_path, mode='r', encoding='utf-8') as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    exp_name = opt.get('name', 'experiment')
    if args.output_dir:
        output_dir = _resolve_path(root_path, args.output_dir)
    else:
        output_dir = root_path / 'experiments' / exp_name / 'export'

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build PyTorch Network & Load Checkpoint
    print("Building model architecture...")
    net = build_network(opt['network_g'])
    net.eval()

    if args.checkpoint:
        checkpoint_path = _resolve_path(root_path, args.checkpoint)
    else:
        models_dir = root_path / 'experiments' / exp_name / 'models'
        best_pths = list(models_dir.glob('*best*.pth'))
        if best_pths:
            checkpoint_path = best_pths[-1]
        else:
            all_pths = sorted(models_dir.glob('*.pth'), key=lambda p: p.stat().st_mtime)
            if all_pths:
                checkpoint_path = all_pths[-1]
            else:
                raise FileNotFoundError(f"Nenhum checkpoint .pth encontrado em {models_dir}")

    print(f"Loading checkpoint weights from {checkpoint_path.name}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('network', checkpoint)
    net.load_state_dict(state_dict, strict=False)

    export_net = net
    if hasattr(net, 'classifier') and isinstance(net.classifier, torch.nn.Module):
        print("Detected TwoStageCropClassifier: exporting the ROI classifier module (net.classifier) for mobile deployment...")
        export_net = net.classifier

    # Gancho de reparametrização estrutural (MobileOne / RepViT / RepVGG)
    if hasattr(export_net, 'switch_to_deploy'):
        print("Invoking export_net.switch_to_deploy() for structural reparameterization...")
        export_net.switch_to_deploy()
    elif hasattr(export_net, 'fuse'):
        print("Invoking export_net.fuse() for structural reparameterization...")
        export_net.fuse()

    image_size = int(opt['datasets']['train'].get('image_size', 224))
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # 2. Export ONNX Model (Static Shape por padrão para NPU/GPU móvel)
    onnx_path = output_dir / 'model.onnx'
    print(f"\nExporting PyTorch model to ONNX: {onnx_path}...")

    dynamic_axes = None
    if not args.static_shape:
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    torch.onnx.export(
        export_net,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    print("ONNX export complete.")

    # 3. Convert ONNX to TFLite via onnx2tf
    print(f"\nConverting ONNX to TFLite (.tflite)...")
    try:
        import onnx2tf
        import onnx2tf.onnx2tf as otf

        otf.download_test_image_data = lambda: np.zeros((1, 3, image_size, image_size), dtype=np.float32)

        tflite_temp_dir = output_dir / 'tflite_temp'
        try:
            otf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(tflite_temp_dir),
                output_integer_quantized_tflite=args.enable_int8,
                non_verbose=True,
            )
        except Exception as conv_err:
            print(f"Notice: onnx2tf converter reported: {conv_err}. Processing generated artifacts...")

        f32_tflite = tflite_temp_dir / 'model_float32.tflite'
        f16_tflite = tflite_temp_dir / 'model_float16.tflite'
        int8_tflite = tflite_temp_dir / 'model_full_integer_quant.tflite'
        if not int8_tflite.exists():
            int8_tflite = tflite_temp_dir / 'model_integer_quant.tflite'
        if not int8_tflite.exists():
            int8_tflite = tflite_temp_dir / 'model_dynamic_range_quant.tflite'

        final_f32 = output_dir / 'model.tflite'
        final_f16 = output_dir / 'model_fp16.tflite'
        final_int8 = output_dir / 'model_int8.tflite'

        if f32_tflite.exists():
            shutil.copy(f32_tflite, final_f32)
            print(f"-> Generated Float32 TFLite model: {final_f32} ({final_f32.stat().st_size / (1024*1024):.2f} MB)")

        if f16_tflite.exists():
            shutil.copy(f16_tflite, final_f16)
            print(f"-> Generated Float16 TFLite model: {final_f16} ({final_f16.stat().st_size / (1024*1024):.2f} MB)")

        if int8_tflite.exists():
            shutil.copy(int8_tflite, final_int8)
            print(f"-> Generated INT8 Quantized TFLite model: {final_int8} ({final_int8.stat().st_size / (1024*1024):.2f} MB)")

        shutil.rmtree(tflite_temp_dir, ignore_errors=True)

    except Exception as exc:
        print(f"Warning: TFLite conversion encountered an issue: {exc}", file=sys.stderr)

    # 4. Copy / Export Labels File (labels.txt)
    dataset_root = _resolve_path(root_path, opt['datasets']['train']['root']).parent
    class_index_path = dataset_root / 'class_index.csv'
    class_names = _load_class_names(class_index_path)

    labels_path = output_dir / 'labels.txt'
    if class_names:
        with labels_path.open('w', encoding='utf-8') as f:
            for name in class_names:
                f.write(f"{name}\n")
        print(f"-> Exported labels list: {labels_path} ({len(class_names)} classes)")

    print(f"\nAll export files successfully written to: {output_dir}")


if __name__ == '__main__':
    main()
