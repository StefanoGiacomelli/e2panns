"""Model Loader for Multi-Model SED System
===========================================
Loads and initializes EPANNs, CED, and CLAP models for inference.

Supports:
- Fine-tuned checkpoints (.pt or .ckpt format)
- Pretrained AudioSet checkpoints (automatic fallback)
- Automatic sample rate detection per model

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn


# Model configurations
MODEL_CONFIGS = {
    'epanns': {
        'sample_rate': 32000,
        'class_index': 322,
        'pretrained_path': 'models/epanns/checkpoint_closeto_.44.pt'
    },
    'ced': {
        'sample_rate': 16000,
        'class_index': 322,
        'pretrained_path': 'models/ced/audiotransformer_base_mAP_4999.pt'
    },
    'clap': {
        'sample_rate': 48000,
        'class_index': 322,
        'pretrained_path': 'models/clap/630k-audioset-fusion-best.pt'
    }
}


def load_inference_model(model_name: str, 
                        checkpoint_path: Optional[str] = None,
                        device: str = 'cpu',
                        project_root: Optional[str] = None) -> Tuple[nn.Module, int, int]:
    """
    Load a model for inference.
    
    Args:
        model_name: Model name ('epanns', 'ced', 'clap')
        checkpoint_path: Path to checkpoint file (.pt or .ckpt). If None, uses pretrained.
        device: Device to load model on ('cpu' or 'cuda')
        project_root: Root directory of the project (for relative paths)
    
    Returns:
        Tuple of (model, target_sr, class_index)
        - model: Loaded PyTorch model in eval mode
        - target_sr: Target sampling rate for this model
        - class_index: Index of Emergency Vehicle class
    
    Raises:
        ValueError: If model_name is invalid
        FileNotFoundError: If checkpoint not found
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    target_sr = config['sample_rate']
    class_index = config['class_index']
    
    logging.info(f"Loading model: {model_name} (sr={target_sr}Hz, class_idx={class_index})")
    
    # Determine project root
    if project_root is None:
        # Try to find it from current file location
        current_file = Path(__file__).resolve()
        # sed_system/core/model_loader.py -> go up 2 levels
        project_root = current_file.parent.parent.parent
    
    # Initialize model architecture
    if model_name == 'epanns':
        model = _load_epanns_model(checkpoint_path, config, device, project_root)
    elif model_name == 'ced':
        model = _load_ced_model(checkpoint_path, config, device, project_root)
    elif model_name == 'clap':
        model = _load_clap_model(checkpoint_path, config, device, project_root)
    
    model.eval()
    logging.info(f"Model loaded successfully: {model_name}")
    
    return model, target_sr, class_index


def _load_epanns_model(checkpoint_path: Optional[str], 
                       config: dict, 
                       device: str,
                       project_root: Path) -> nn.Module:
    """
    Load EPANNs model.
    
    Supports both:
    - Pretrained checkpoints (direct state_dict with 'model.' prefix)
    - Lightning finetuned checkpoints (with 'pytorch_model_state_dict' key)
    """
    from models.epanns.model import EPANNs
    
    model = EPANNs(sample_rate=config['sample_rate'])
    
    # Load checkpoint
    if checkpoint_path is None:
        # Use pretrained
        checkpoint_path = project_root / config['pretrained_path']
        logging.info(f"Using pretrained checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(checkpoint_path)
        logging.info(f"Using finetuned checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect checkpoint format
    if isinstance(checkpoint, dict) and 'pytorch_model_state_dict' in checkpoint:
        # Lightning checkpoint format (finetuned)
        logging.info("Detected Lightning checkpoint format")
        state_dict = checkpoint['pytorch_model_state_dict']
        
        # Log metadata if available
        if 'epoch' in checkpoint:
            logging.info(f"  Epoch: {checkpoint['epoch']}")
        if 'monitor_value' in checkpoint:
            logging.info(f"  Monitor metric: {checkpoint.get('monitor_metric', 'N/A')} = {checkpoint['monitor_value']:.4f}")
        
        # Remove 'model.' prefix from keys (Lightning wrapper)
        state_dict_clean = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                state_dict_clean[key[6:]] = value  # Remove 'model.' prefix
            else:
                state_dict_clean[key] = value
        
        # Load into model.model (Cnn14Pruned)
        model.model.load_state_dict(state_dict_clean, strict=True)
        logging.info(f"Finetuned checkpoint loaded successfully")
    else:
        # Pretrained checkpoint format (direct state_dict)
        logging.info("Detected pretrained checkpoint format")
        model.load_pretrained(str(checkpoint_path))
        logging.info(f"Pretrained checkpoint loaded successfully")
    
    return model.to(device)


def _load_ced_model(checkpoint_path: Optional[str],
                    config: dict,
                    device: str,
                    project_root: Path) -> nn.Module:
    """Load CED model."""
    from models.ced.model import CEDBase
    
    model = CEDBase(sample_rate=config['sample_rate'])
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = project_root / config['pretrained_path']
        logging.info(f"Using pretrained checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Use model's load_pretrained method
    model.load_pretrained(str(checkpoint_path))
    logging.info(f"CED checkpoint loaded successfully")
    
    return model.to(device)


def _load_clap_model(checkpoint_path: Optional[str],
                     config: dict,
                     device: str,
                     project_root: Path) -> nn.Module:
    """Load CLAP model."""
    from models.clap.model import CLAP
    
    model = CLAP(sample_rate=config['sample_rate'])
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = project_root / config['pretrained_path']
        logging.info(f"Using pretrained checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Use model's load_pretrained method
    model.load_pretrained(str(checkpoint_path))
    logging.info(f"CLAP checkpoint loaded successfully")
    
    return model.to(device)


def get_model_config(model_name: str) -> dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Model name
    
    Returns:
        Dictionary with model configuration
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return MODEL_CONFIGS[model_name].copy()

