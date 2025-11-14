#!/usr/bin/env python3
"""
Evaluate trained model on test positions.

Computes metrics like:
- Move prediction accuracy
- Value prediction error
- Tactical puzzle solving rate
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import create_model
from datasets import PGNDataset
from utils import load_config, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                        help='Path to model config')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test dataset NPZ')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Report top-k move accuracy')
    return parser.parse_args()


def evaluate_model(model, dataset, device, top_k=3):
    """Evaluate model on dataset."""
    model.eval()
    
    correct_top1 = 0
    correct_topk = 0
    value_errors = []
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False
    )
    
    with torch.no_grad():
        for planes, move_indices, results in dataloader:
            planes = planes.to(device)
            move_indices = move_indices.to(device)
            results = results.to(device)
            
            # Predict
            policy_logits, value_pred = model(planes)
            
            # Top-1 accuracy
            pred_moves = policy_logits.argmax(dim=-1)
            correct_top1 += (pred_moves == move_indices).sum().item()
            
            # Top-k accuracy
            _, topk_pred = policy_logits.topk(top_k, dim=-1)
            correct_topk += (topk_pred == move_indices.unsqueeze(-1)).any(dim=-1).sum().item()
            
            # Value error
            value_error = torch.abs(value_pred.squeeze() - results)
            value_errors.extend(value_error.cpu().numpy().tolist())
    
    total = len(dataset)
    
    metrics = {
        'top1_accuracy': correct_top1 / total,
        'topk_accuracy': correct_topk / total,
        'mean_value_error': np.mean(value_errors),
        'median_value_error': np.median(value_errors),
    }
    
    return metrics


def main():
    args = parse_args()
    
    # Setup
    logger = setup_logger('evaluate')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = load_config(args.config)
    model_config = config['model']
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = create_model(model_config)
    
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_dataset = PGNDataset(args.test_data)
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, test_dataset, device, top_k=args.top_k)
    
    # Report
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"Top-1 Move Accuracy: {metrics['top1_accuracy']:.2%}")
    logger.info(f"Top-{args.top_k} Move Accuracy: {metrics['topk_accuracy']:.2%}")
    logger.info(f"Mean Value Error: {metrics['mean_value_error']:.4f}")
    logger.info(f"Median Value Error: {metrics['median_value_error']:.4f}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
