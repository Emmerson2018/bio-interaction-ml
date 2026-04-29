import os

def scandir(dir_path, suffix=None, recursive=False):
    """Varre um diretório em busca de arquivos com um sufixo específico."""
    for root, _, files in os.walk(dir_path):
        for file in files:
            if suffix is None or file.endswith(suffix):
                yield os.path.join(root, file)
        if not recursive:
            break
