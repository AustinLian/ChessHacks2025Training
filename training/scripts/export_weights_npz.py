#!/usr/bin/env python3
"""
Export trained model weights to format usable by C++ engine.

Converts PyTorch model to NPZ format with raw weight arrays.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import create_model
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Export model weights for C++ engine')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PyTorch model checkpoint')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                        help='Path to model config')
    parser.add_argument('--output', type=str, required=True,
                        help='Output NPZ file path')
    return parser.parse_args()


def export_weights(model, output_path):
    """
    Export model weights to NPZ format for C++ inference.
    
    Format:
    - Each layer's weights and biases are saved as separate arrays
    - Naming convention: layer_name/weight, layer_name/bias
    """
    state_dict = model.state_dict()
    
    # Convert tensors to numpy arrays
    weight_dict = {}
    for key, tensor in state_dict.items():
        # Convert to numpy and ensure C-contiguous
        array = tensor.cpu().numpy().copy()
        weight_dict[key] = array
        print(f"{key}: {array.shape} {array.dtype}")
    
    # Save to NPZ
    np.savez_compressed(output_path, **weight_dict)
    print(f"\nExported {len(weight_dict)} arrays to {output_path}")
    
    # Save model architecture info
    arch_info = {
        'input_planes': model.input_planes,
        'num_blocks': model.num_blocks,
        'channels': model.channels,
    }
    
    arch_path = output_path.parent / 'engine_config.json'
    with open(arch_path, 'w') as f:
        json.dump(arch_info, f, indent=2)
    print(f"Saved architecture config to {arch_path}")
    
    # Save version info
    version_path = output_path.parent / 'VERSION'
    with open(version_path, 'w') as f:
        f.write("1.0.0\n")
    print(f"Saved version info to {version_path}")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    model_config = config['model']
    
    print(f"Loading model from {args.model}")
    
    # Create model
    model = create_model(model_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model architecture: {model_config}")
    
    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_weights(model, output_path)
    
    print("\nExport complete!")
    print(f"C++ engine can load weights from: {output_path}")


if __name__ == '__main__':
    main()
