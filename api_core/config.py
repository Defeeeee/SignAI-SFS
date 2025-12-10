import os
import platform
import argparse
import torch
import logging

logger = logging.getLogger(__name__)

def get_model_args():
    """
    Determines the configuration, dictionary, and weights paths based on the operating system.
    Also detects the best available device (CUDA, MPS, or CPU).
    """
    # Determine base directory (project root)
    # Assuming this file is in <project_root>/api_core/config.py
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Set paths based on system
    if platform.system() == 'Darwin':  # Mac
        # Keep the absolute path logic for Mac as in original code, or use relative if preferred.
        # The original code used absolute paths for Mac, so we'll stick to that if possible,
        # but using base_dir is more robust. Let's use base_dir which is absolute.
        config_path = os.path.join(base_dir, 'configs', 'phoenix2014-T.yaml')
        dict_path = os.path.join(base_dir, 'preprocess', 'phoenix2014-T', 'gloss_dict.npy')
        weights_path = os.path.join(base_dir, 'best_checkpoints', 'phoenix2014-T_dev_17.66_test_18.71.pt')
    else:  # Windows/Linux
        config_path = os.path.join(base_dir, 'configs', 'phoenix2014-T.yaml')
        dict_path = os.path.join(base_dir, 'preprocess', 'phoenix2014-T', 'gloss_dict.npy')
        weights_path = os.path.join(base_dir, 'best_checkpoints', 'phoenix2014-T_dev_17.66_test_18.71.pt')
        
    # Determine device
    device_str = 'cpu'
    if torch.cuda.is_available():
        device_str = 'cuda:0'
        logger.info("CUDA is available. Using GPU.")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_str = 'mps'
        logger.info("MPS (Apple Silicon) is available. Using MPS.")
    else:
        logger.info("Using CPU.")
        
    # Set number of threads for CPU inference on Mac/ARM
    if device_str == 'cpu' and platform.system() == 'Darwin':
        # On ARM/Mac, using too many threads can hurt performance due to overhead
        # 4 is usually a sweet spot for M1/M2/M3 chips
        torch.set_num_threads(4)
        logger.info("Set torch num threads to 4 for CPU optimization")

    args = argparse.Namespace(
        config=config_path,
        dict_path=dict_path,
        weights=weights_path,
        device=device_str
    )
    return args
