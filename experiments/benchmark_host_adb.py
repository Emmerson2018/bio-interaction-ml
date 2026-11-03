import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def benchmark_pytorch_model(model, input_shape=(1, 3, 224, 224), device='cpu', warmup=20, runs=100):
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    x = torch.randn(*input_shape, device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return {
        'device': device,
        'mean_ms': float(np.mean(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p90_ms': float(np.percentile(times, 90)),
        'p99_ms': float(np.percentile(times, 99)),
        'fps': float(1000.0 / np.mean(times)),
    }


def benchmark_tflite_model(tflite_path, input_shape=(1, 3, 224, 224), num_threads=4, warmup=20, runs=100):
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            return None

    try:
        interp = tflite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()

        in_dtype = input_details[0]['dtype']
        if in_dtype == np.int8:
            dummy = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
        else:
            dummy = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(warmup):
            interp.set_tensor(input_details[0]['index'], dummy)
            interp.invoke()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            interp.set_tensor(input_details[0]['index'], dummy)
            interp.invoke()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        return {
            'num_threads': num_threads,
            'mean_ms': float(np.mean(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p90_ms': float(np.percentile(times, 90)),
            'p99_ms': float(np.percentile(times, 99)),
            'fps': float(1000.0 / np.mean(times)),
        }
    except Exception as e:
        print(f"Erro no benchmark TFLite: {e}")
        return None


def check_adb_devices():
    adb_path = shutil.which('adb')
    if not adb_path:
        return []
    try:
        res = subprocess.run(['adb', 'devices'], capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split('\n')[1:]
        devices = [line.split('\t')[0] for line in lines if '\tdevice' in line]
        return devices
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description='Benchmark Host & Android via ADB.')
    parser.add_argument('--export-dir', type=str, required=True, help='Path to export directory containing model.tflite / model_int8.tflite')
    args = parser.parse_args()

    export_path = Path(args.export_dir)
    print(f"\n==========================================")
    print(f"Executando Benchmark de Inferência para: {export_path}")
    print(f"==========================================")

    results = {}
    f32_path = export_path / 'model.tflite'
    f16_path = export_path / 'model_fp16.tflite'
    int8_path = export_path / 'model_int8.tflite'

    if f32_path.exists():
        results['tflite_f32'] = benchmark_tflite_model(f32_path)
        if results['tflite_f32']:
            print(f"TFLite Float32 (Host CPU 4T): {results['tflite_f32']['mean_ms']:.2f} ms ({results['tflite_f32']['fps']:.1f} FPS)")

    if f16_path.exists():
        results['tflite_f16'] = benchmark_tflite_model(f16_path)
        if results['tflite_f16']:
            print(f"TFLite Float16 (Host CPU 4T): {results['tflite_f16']['mean_ms']:.2f} ms ({results['tflite_f16']['fps']:.1f} FPS)")

    if int8_path.exists():
        results['tflite_int8'] = benchmark_tflite_model(int8_path)
        if results['tflite_int8']:
            print(f"TFLite INT8 (Host CPU 4T): {results['tflite_int8']['mean_ms']:.2f} ms ({results['tflite_int8']['fps']:.1f} FPS)")

    # ADB Device Check
    devices = check_adb_devices()
    if devices:
        print(f"\nDispositivo Android detectado via ADB: {devices[0]}")
    else:
        print(f"\nNenhum dispositivo Android físico conectado via ADB (Hardware Guardrail ativo).")

    out_file = export_path / 'benchmark_summary.json'
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark gravado em: {out_file}")


if __name__ == '__main__':
    main()
