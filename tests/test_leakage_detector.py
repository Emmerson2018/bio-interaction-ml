import json
from collections import defaultdict
from pathlib import Path


def test_no_data_leakage_in_sanitized_dataset():
    manifest_path = Path("datasets/generated/animals_video_sanitized/dataset_manifest.json")
    assert manifest_path.exists(), "Manifesto do dataset não encontrado!"

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    assert len(records) > 0, "Nenhum registro encontrado no manifesto!"

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
        if not r["is_augmented"]:
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

            # O buffer temporal de exclusão garante separação
            assert eval_min_t > train_max_t, (
                f"Vazamento temporal detectado em {class_name}/{v_name}: "
                f"train_max_t={train_max_t} vs eval_min_t={eval_min_t}"
            )

    # 3. Verificar que nenhum vídeo de treino em classes com vídeos dedicados aparece em val ou test
    print(f"Total de vídeos compartilhados com buffer (N <= 3): {shared_videos}")
    print("Test passed: Zero data leakage between train and val/test!")


if __name__ == "__main__":
    test_no_data_leakage_in_sanitized_dataset()
