import json

import pytest

from base_tool.export_mobile import load_content_ids, resolve_deployment_ids


def test_deployment_ids_are_resolved_from_app_database(tmp_path):
    content_db = tmp_path / 'animais.json'
    content_db.write_text(
        json.dumps([{'id': 'capivara'}, {'id': 'sapo_flecha'}]),
        encoding='utf-8',
    )

    resolved = resolve_deployment_ids(
        ['capivara', 'sapo', 'unknown'],
        load_content_ids(content_db),
        {'sapo': 'sapo_flecha'},
    )

    assert resolved == ['capivara', 'sapo_flecha', 'unknown']


def test_deployment_rejects_id_missing_from_app_database(tmp_path):
    content_db = tmp_path / 'animais.json'
    content_db.write_text(json.dumps([{'id': 'capivara'}]), encoding='utf-8')

    with pytest.raises(ValueError, match='does not exist'):
        resolve_deployment_ids(['onca_pintada'], load_content_ids(content_db))


def test_deployment_rejects_duplicate_resolved_ids():
    with pytest.raises(ValueError, match='must be unique'):
        resolve_deployment_ids(
            ['sapo', 'sapo_flecha'],
            {'sapo_flecha'},
            {'sapo': 'sapo_flecha'},
        )
