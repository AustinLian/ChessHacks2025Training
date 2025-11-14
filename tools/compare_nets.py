#!/usr/bin/env python3
"""
Compare two neural network models on test positions.

Useful for A/B testing and model selection.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.models import create_model
from training.datasets import PGNDataset
from training.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Compare two models')
    parser.add_argument('--model-a', type=str, required=True,
                        help='Path to model A checkpoint')
    parser.add_argument('--model-b', type=str, required=True,
                        help='Path to model B checkpoint')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                        help='Path to model config')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test dataset')
    return parser.parse_args()


def evaluate_model(model, dataset, device):
    """Evaluate model on dataset."""
    model.eval()
    
    correct = 0
    value_errors = []
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False
    )
    
    with torch.no_grad():
        for planes, move_indices, results in dataloader:
            planes = planes.to(device)
            move_indices = move_indices.to(device)
            results = results.to(device)
            
            policy_logits, value_pred = model(planes)
            
            # Accuracy
            pred_moves = policy_logits.argmax(dim=-1)
            correct += (pred_moves == move_indices).sum().item()
            
            # Value error
            errors = torch.abs(value_pred.squeeze() - results)
            value_errors.extend(errors.cpu().numpy().tolist())
    
    accuracy = correct / len(dataset)
    mean_value_error = np.mean(value_errors)
    
    return accuracy, mean_value_error


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    # Load models
    print("Loading models...")
    model_a = create_model(config['model'])
    model_b = create_model(config['model'])
    
    checkpoint_a = torch.load(args.model_a, map_location='cpu')
    checkpoint_b = torch.load(args.model_b, map_location='cpu')
    
    model_a.load_state_dict(checkpoint_a['model_state_dict'])
    model_b.load_state_dict(checkpoint_b['model_state_dict'])
    
    model_a = model_a.to(device)
    model_b = model_b.to(device)
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_dataset = PGNDataset(args.test_data)
    print(f"Test set size: {len(test_dataset)}")
    
    # Evaluate both models
    print("\nEvaluating Model A...")
    acc_a, val_err_a = evaluate_model(model_a, test_dataset, device)
    
    print("Evaluating Model B...")
    acc_b, val_err_b = evaluate_model(model_b, test_dataset, device)
    
    # Compare
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"{'Metric':<30} {'Model A':<15} {'Model B':<15} {'Winner'}")
    print("-" * 60)
    
    acc_winner = "A" if acc_a > acc_b else ("B" if acc_b > acc_a else "Tie")
    val_winner = "A" if val_err_a < val_err_b else ("B" if val_err_b < val_err_a else "Tie")
    
    print(f"{'Move Accuracy':<30} {acc_a:<15.4f} {acc_b:<15.4f} {acc_winner}")
    print(f"{'Value Error':<30} {val_err_a:<15.4f} {val_err_b:<15.4f} {val_winner}")
    
    print("=" * 60)
    
    # Overall winner
    if acc_winner == val_winner and acc_winner != "Tie":
        print(f"\n🏆 Overall Winner: Model {acc_winner}")
    else:
        print("\n⚖️  Mixed results - further evaluation recommended")


if __name__ == '__main__':
    main()
