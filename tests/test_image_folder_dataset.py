from PIL import Image

from base_tool.data.image_folder_dataset import ImageFolderClassificationDataset


def _write_image(path):
    image = Image.new('RGB', (16, 16), color=(120, 80, 40))
    image.save(path)


def test_image_folder_preserves_sorted_class_order_and_metadata(tmp_path):
    sapo = tmp_path / 'sapo'
    capivara = tmp_path / 'capivara'
    sapo.mkdir()
    capivara.mkdir()
    _write_image(sapo / 'sapo_001.png')
    _write_image(capivara / 'capivara_001.png')

    dataset = ImageFolderClassificationDataset(
        {
            'root': str(tmp_path),
            'augment': False,
            'image_size': 32,
            'network_g': {
                'backbone': 'resnet18',
                'weights': None,
                'preprocessing': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
            },
        }
    )

    assert dataset.classes == ['capivara', 'sapo']
    assert dataset.class_to_idx == {'capivara': 0, 'sapo': 1}
    assert dataset.idx_to_class == {0: 'capivara', 1: 'sapo'}

    metadata = dataset.get_metadata()
    assert metadata['class_to_idx'] == {'capivara': 0, 'sapo': 1}
    assert metadata['idx_to_class'] == {'0': 'capivara', '1': 'sapo'}
    assert metadata['preprocessing']['input_size'] == 32


def test_image_folder_output_shape(tmp_path):
    class_dir = tmp_path / 'capivara'
    class_dir.mkdir()
    _write_image(class_dir / 'sample.png')

    dataset = ImageFolderClassificationDataset(
        {
            'root': str(tmp_path),
            'augment': False,
            'image_size': 32,
            'network_g': {
                'backbone': 'resnet18',
                'weights': None,
                'preprocessing': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
            },
        }
    )

    item = dataset[0]
    assert item['x'].shape == (3, 32, 32)
    assert item['y'] == 0
