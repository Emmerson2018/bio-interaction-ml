from pathlib import Path

import pytest

from base_tool.synthetic.generate_blender_dataset import (
    _discover_models,
    _load_class_mapping,
    _parse_last_json_line,
    _write_class_index,
)


def test_discovers_supported_blend_and_blend1_files(tmp_path):
    source_root = tmp_path / "assets" / "blender_models"
    (source_root / "capivara").mkdir(parents=True)
    (source_root / "sapo").mkdir(parents=True)
    (source_root / "capivara" / "Capivara.blend1").write_bytes(b"blend backup")
    (source_root / "sapo" / "Sapo.blend").write_bytes(b"blend")

    rows = _discover_models(source_root, {"capivara": "capivara", "sapo": "sapo"})

    assert [row["canonical_class"] for row in rows] == ["capivara", "sapo"]
    assert [row["extension"] for row in rows] == [".blend1", ".blend"]
    assert all(row["status"] == "ok" for row in rows)
    assert all(Path(row["path"]).exists() for row in rows)


def test_writes_deterministic_class_index(tmp_path):
    output_path = tmp_path / "class_index.csv"

    _write_class_index(output_path, ["bicho_preguica", "boto_cor_de_rosa", "capivara"])

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "class_name,class_index",
        "bicho_preguica,0",
        "boto_cor_de_rosa,1",
        "capivara,2",
    ]


def test_rejects_duplicate_canonical_class_ids(tmp_path):
    mapping_path = tmp_path / "mapping.yml"
    mapping_path.write_text(
        "class_mapping:\n"
        "  A: capivara\n"
        "  B: capivara\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate canonical class ids"):
        _load_class_mapping(tmp_path, {"class_mapping_file": "mapping.yml"})


def test_reports_missing_model_file(tmp_path):
    source_root = tmp_path / "assets" / "blender_models"
    source_root.mkdir(parents=True)

    rows = _discover_models(source_root, {"capivara": "capivara"})

    assert rows[0]["canonical_class"] == "capivara"
    assert rows[0]["status"] == "missing_model_file"
    assert rows[0]["path"] is None


def test_parse_last_json_line_ignores_blender_logs():
    stdout = 'Read blend: "model.blend"\n{"status": "ok", "mesh_count": 1}\nError: Not freed memory blocks\n'

    assert _parse_last_json_line(stdout) == {"status": "ok", "mesh_count": 1}
