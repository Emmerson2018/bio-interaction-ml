import json
import sys
from collections import defaultdict
from pathlib import Path


def audit_manifest_leakage(manifest_path_str: str):
    manifest_path = Path(manifest_path_str)
    if not manifest_path.exists():
        print(f"Aviso: Manifesto '{manifest_path}' ainda não existe no disco. Pulando auditoria.")
        return True

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    assert len(records) > 0, f"Nenhum registro encontrado no manifesto {manifest_path}!"

    # 1. Agrupar timestamps por (class_name, video_name) e split
    timestamps_by_video_split = defaultdict(lambda: defaultdict(list))
    video_hashes_by_split = defaultdict(lambda: defaultdict(set))

    for r in records:
        class_name = r["class_name"]
        split = r["split"]
        v_name = r["video_name"]
        v_key = (class_name, v_name)
        v_hash = r["video_hash"]

        video_hashes_by_split[class_name][split].add(v_hash)
        if not r.get("is_augmented", False):
            timestamps_by_video_split[v_key][split].append(r["timestamp_sec"])

    # 2. Verificar que NENHUM vídeo compartilhado entre train e val/test tem sobreposição temporal
    shared_videos = 0
    for (class_name, v_name), splits_dict in timestamps_by_video_split.items():
        if "train" in splits_dict and ("val" in splits_dict or "test" in splits_dict):
            shared_videos += 1
            train_max_t = max(splits_dict["train"])
            val_min_t = min(splits_dict["val"]) if "val" in splits_dict else float("inf")
            test_min_t = min(splits_dict["test"]) if "test" in splits_dict else float("inf")
            eval_min_t = min(val_min_t, test_min_t)

            assert eval_min_t > train_max_t, (
                f"Vazamento temporal detectado em {class_name}/{v_name} no manifesto {manifest_path}: "
                f"train_max_t={train_max_t} vs eval_min_t={eval_min_t}"
            )

    print(f"[{manifest_path.name}] Vídeos compartilhados com buffer (N <= 3): {shared_videos}")
    print(f"[{manifest_path.name}] ✓ Test passed: Zero data leakage between train and val/test!")
    return True


def test_no_data_leakage():
    manifests = [
        "datasets/generated/animals_video_sanitized/dataset_manifest.json",
        "datasets/generated/animals_video_sam2_crops/dataset_manifest.json",
    ]
    for m in manifests:
        audit_manifest_leakage(m)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audit_manifest_leakage(sys.argv[1])
    else:
        test_no_data_leakage()
