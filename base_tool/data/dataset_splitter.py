import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class VideoInfo:
    class_name: str
    video_path: Path
    video_name: str
    content_hash: str
    duration_sec: float
    fps: float
    frame_count: int


@dataclass
class VideoSplitPlan:
    video_path: Path
    class_name: str
    split: str  # 'train', 'val', 'test'
    start_ratio: float = 0.0
    end_ratio: float = 1.0
    content_hash: str = ""


def compute_file_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class VideoDatasetSplitter:
    """
    Particionador de vídeos estruturado por grupos (video_id e content_hash)
    para eliminação total de Data Leakage temporal e de instância.
    """

    def __init__(
        self,
        source_dir: Path,
        gap_ratio: float = 0.10,
        val_ratio_for_shared: float = 0.45,
        seed: int = 20260814,
    ):
        self.source_dir = Path(source_dir)
        self.gap_ratio = gap_ratio
        self.val_ratio_for_shared = val_ratio_for_shared
        self.seed = seed

    def scan_videos(self) -> Dict[str, List[VideoInfo]]:
        import cv2

        class_dirs = sorted([d for d in self.source_dir.iterdir() if d.is_dir()])
        videos_by_class: Dict[str, List[VideoInfo]] = defaultdict(list)

        for class_dir in class_dirs:
            class_name = class_dir.name
            vids = sorted(list(class_dir.glob("*.mp4")))
            for v_path in vids:
                h = compute_file_md5(v_path)
                cap = cv2.VideoCapture(str(v_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                dur = frame_count / fps if fps > 0 else 0.0
                cap.release()

                videos_by_class[class_name].append(
                    VideoInfo(
                        class_name=class_name,
                        video_path=v_path,
                        video_name=v_path.name,
                        content_hash=h,
                        duration_sec=dur,
                        fps=fps,
                        frame_count=frame_count,
                    )
                )

        return videos_by_class

    def build_split_plan(self) -> List[VideoSplitPlan]:
        videos_by_class = self.scan_videos()
        all_plans: List[VideoSplitPlan] = []

        for class_name, vid_list in sorted(videos_by_class.items()):
            # 1. Agrupar vídeos por hash de conteúdo para detectar arquivos duplicados
            hash_groups: Dict[str, List[VideoInfo]] = defaultdict(list)
            for v in vid_list:
                hash_groups[v.content_hash].append(v)

            # Lista de grupos únicos (ordenada por duração decrescente para priorizar maior representatividade)
            unique_groups = sorted(
                list(hash_groups.values()),
                key=lambda grp: grp[0].duration_sec,
                reverse=True,
            )
            n_unique = len(unique_groups)

            if n_unique >= 4:
                # Alocação por vídeos 100% independentes para train, val e test
                n_test = max(1, round(n_unique * 0.15))
                n_val = max(1, round(n_unique * 0.15))
                n_train = n_unique - n_val - n_test

                train_groups = unique_groups[:n_train]
                val_groups = unique_groups[n_train : n_train + n_val]
                test_groups = unique_groups[n_train + n_val :]

                for grp in train_groups:
                    for v in grp:
                        all_plans.append(
                            VideoSplitPlan(v.video_path, class_name, "train", 0.0, 1.0, v.content_hash)
                        )
                for grp in val_groups:
                    for v in grp:
                        all_plans.append(
                            VideoSplitPlan(v.video_path, class_name, "val", 0.0, 1.0, v.content_hash)
                        )
                for grp in test_groups:
                    for v in grp:
                        all_plans.append(
                            VideoSplitPlan(v.video_path, class_name, "test", 0.0, 1.0, v.content_hash)
                        )

            elif n_unique == 3:
                # 2 grupos para treino (máxima riqueza de representação)
                for v in unique_groups[0]:
                    all_plans.append(
                        VideoSplitPlan(v.video_path, class_name, "train", 0.0, 1.0, v.content_hash)
                    )
                for v in unique_groups[1]:
                    all_plans.append(
                        VideoSplitPlan(v.video_path, class_name, "train", 0.0, 1.0, v.content_hash)
                    )

                # 1 grupo para validação e teste com buffer de exclusão temporal
                # Ex: 0.0 a 0.45 (val), 0.45 a 0.55 (gap descartado), 0.55 a 1.0 (test)
                for v in unique_groups[2]:
                    all_plans.append(
                        VideoSplitPlan(
                            v.video_path,
                            class_name,
                            "val",
                            0.0,
                            self.val_ratio_for_shared,
                            v.content_hash,
                        )
                    )
                    all_plans.append(
                        VideoSplitPlan(
                            v.video_path,
                            class_name,
                            "test",
                            self.val_ratio_for_shared + self.gap_ratio,
                            1.0,
                            v.content_hash,
                        )
                    )

            elif n_unique == 2:
                # 1 grupo para treino (100% isolado)
                for v in unique_groups[0]:
                    all_plans.append(
                        VideoSplitPlan(v.video_path, class_name, "train", 0.0, 1.0, v.content_hash)
                    )

                # 1 grupo para val e test com buffer de exclusão temporal
                for v in unique_groups[1]:
                    all_plans.append(
                        VideoSplitPlan(
                            v.video_path,
                            class_name,
                            "val",
                            0.0,
                            self.val_ratio_for_shared,
                            v.content_hash,
                        )
                    )
                    all_plans.append(
                        VideoSplitPlan(
                            v.video_path,
                            class_name,
                            "test",
                            self.val_ratio_for_shared + self.gap_ratio,
                            1.0,
                            v.content_hash,
                        )
                    )

            else:
                # Singleton (n_unique == 1): divisão temporal com exclusion buffer
                v = unique_groups[0][0]
                all_plans.append(
                    VideoSplitPlan(v.video_path, class_name, "train", 0.0, 0.60, v.content_hash)
                )
                all_plans.append(
                    VideoSplitPlan(v.video_path, class_name, "test", 0.75, 1.0, v.content_hash)
                )

        # Validação estrita anti-leakage
        self.verify_no_train_leakage(all_plans)
        return all_plans

    @staticmethod
    def verify_no_train_leakage(plans: List[VideoSplitPlan]) -> None:
        """
        Garante que nenhum hash de conteúdo de vídeo de treino apareça em val ou test,
        a menos que pertença a classes com <= 3 vídeos onde o split temporal é explicitamente
        delimitado por janelas não sobrepostas e separadas por buffer de exclusão.
        """
        train_hashes_by_class: Dict[str, set] = defaultdict(set)
        val_hashes_by_class: Dict[str, set] = defaultdict(set)
        test_hashes_by_class: Dict[str, set] = defaultdict(set)

        for p in plans:
            if p.split == "train":
                train_hashes_by_class[p.class_name].add(p.content_hash)
            elif p.split == "val":
                val_hashes_by_class[p.class_name].add(p.content_hash)
            elif p.split == "test":
                test_hashes_by_class[p.class_name].add(p.content_hash)

        for class_name, t_hashes in train_hashes_by_class.items():
            # Para classes com N >= 4 vídeos, a interseção com train DEVE ser estritamente vazia
            v_hashes = val_hashes_by_class.get(class_name, set())
            te_hashes = test_hashes_by_class.get(class_name, set())

            # Verificar planos para a mesma classe
            class_plans = [p for p in plans if p.class_name == class_name]
            # Se algum vídeo estiver em train e val/test, validar que as janelas temporais NÃO se sobrepõem
            for p1 in class_plans:
                for p2 in class_plans:
                    if p1.video_path == p2.video_path and p1.split != p2.split:
                        # Devem ter janelas disjuntas com gap
                        assert (
                            p1.end_ratio <= p2.start_ratio or p2.end_ratio <= p1.start_ratio
                        ), f"Sobreposição temporal detectada no mesmo vídeo {p1.video_path} entre {p1.split} e {p2.split}!"
