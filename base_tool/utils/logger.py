import logging
import os
import time

def get_root_logger(logger_name='base_tool', log_level=logging.INFO, log_file=None):
    """Configura o logger principal."""
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    if log_file:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger

def get_env_info():
    """Retorna informações básicas do ambiente para o log inicial."""
    import torch
    return f"PyTorch: {torch.__version__} | GPU: {torch.cuda.is_available()}"
