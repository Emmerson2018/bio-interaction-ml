import copy
import torch
from base_tool.archs import build_network
from base_tool.models import build_model
from base_tool.utils.options import parse_options


def smoke_test_config(opt_path):
    print(f"\n---> Smoke test para configuração: {opt_path}")
    opt = parse_options(opt_path, is_train=True)
    opt['train']['total_epochs'] = 1
    num_classes = int(opt['network_g'].get('num_classes', 16))

    model = build_model(opt)
    dummy_x = torch.randn(2, 3, 224, 224)
    dummy_y = torch.tensor([0, 1])

    # 1. Feed & Optimize
    model.feed_data({'x': dummy_x, 'y': dummy_y})
    model.optimize_parameters(current_iter=1)
    losses = model.get_current_losses()
    print(f"  Step de treino concluído com sucesso. Loss: {losses.get('loss'):.4f}, Acc: {losses.get('acc'):.4f}")

    # 2. Test / Eval
    model.test()
    visuals = model.get_current_visuals()
    assert visuals['prediction'].shape == (2, num_classes), f"Formato inesperado: {visuals['prediction'].shape} vs esperado (2, {num_classes})"
    print(f"  Inferência concluída com sucesso. Shape: {visuals['prediction'].shape}")


def main():
    print("==========================================")
    print("Iniciando bateria de Smoke Tests para os Experimentos")
    print("==========================================")

    configs = [
        'options/exp1_resnet18_baseline.yml',
        'options/exp2_resnet18_yolo_crop.yml',
        'options/exp3_resnet18_ca.yml',
        'options/exp_mobilenetv4_small.yml',
        'options/exp_repvit_m0_9.yml',
        'options/exp_loss_ce.yml',
        'options/exp_loss_focal.yml',
        'options/exp_mobilenetv4_sam2_crops.yml',
    ]

    for c in configs:
        smoke_test_config(c)

    print("\n==========================================")
    print(f"Todos os {len(configs)} Experimentos passaram nos Smoke Tests com 100% de sucesso!")
    print("==========================================")


if __name__ == '__main__':
    main()
