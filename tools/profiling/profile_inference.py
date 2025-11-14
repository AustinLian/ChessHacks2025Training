#!/usr/bin/env python3
"""
Profile neural network inference performance.

Measures forward pass latency and throughput.
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.models import create_model
from training.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Profile NN inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                        help='Path to model config')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations')
    return parser.parse_args()


def profile_inference(model, batch_size, iterations, device):
    """Profile model inference."""
    model.eval()
    
    # Generate random input
    dummy_input = torch.randn(batch_size, 27, 8, 8).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Profile
    times = []
    
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            policy, value = model(dummy_input)
            
            # Ensure computation is complete
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
    }


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config and model
    config = load_config(args.config)
    model = create_model(config['model'])
    
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    
    # Profile
    print("\nProfiling inference...")
    results = profile_inference(model, args.batch_size, args.iterations, device)
    
    # Report
    print("\n" + "=" * 60)
    print("Inference Performance")
    print("=" * 60)
    print(f"Mean latency:     {results['mean']:.2f} ms")
    print(f"Median latency:   {results['median']:.2f} ms")
    print(f"Min latency:      {results['min']:.2f} ms")
    print(f"Max latency:      {results['max']:.2f} ms")
    print(f"Std deviation:    {results['std']:.2f} ms")
    print(f"95th percentile:  {results['p95']:.2f} ms")
    print(f"99th percentile:  {results['p99']:.2f} ms")
    print("=" * 60)
    
    # Throughput
    positions_per_sec = (args.batch_size * args.iterations) / (sum(times := np.array([results['mean']] * args.iterations)) / 1000)
    print(f"\nThroughput: {positions_per_sec:.0f} positions/sec")
    print(f"Time per position: {1000 / positions_per_sec:.2f} ms")


if __name__ == '__main__':
    main()
