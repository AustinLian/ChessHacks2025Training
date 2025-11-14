#!/usr/bin/env python3
"""
Reinforcement learning training using self-play buffer.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import create_model
from datasets import SelfPlayDataset
from utils import (
    load_config,
    setup_logger,
    log_metrics,
    save_checkpoint,
    AverageMeter,
    load_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description='RL training with self-play')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                        help='Path to training config')
    parser.add_argument('--buffer', type=str, required=True,
                        help='Path to self-play buffer NPZ')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to initial model checkpoint')
    parser.add_argument('--output-dir', type=str, default='weights/checkpoints',
                        help='Directory to save checkpoints')
    return parser.parse_args()


def train_iteration(model, dataloader, optimizer, device, logger, value_weight=1.0):
    """Train for one iteration on self-play buffer."""
    model.train()
    
    policy_loss_meter = AverageMeter('policy_loss')
    value_loss_meter = AverageMeter('value_loss')
    total_loss_meter = AverageMeter('total_loss')
    
    for batch_idx, (planes, target_policies, target_values) in enumerate(dataloader):
        planes = planes.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)
        
        # Forward pass
        policy_logits, value_pred = model(planes)
        
        # Policy loss (KL divergence with target policy from search)
        policy_loss = nn.functional.kl_div(
            nn.functional.log_softmax(policy_logits, dim=-1),
            target_policies,
            reduction='batchmean'
        )
        
        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(value_pred.squeeze(), target_values)
        
        # Combined loss
        total_loss = policy_loss + value_weight * value_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update meters
        policy_loss_meter.update(policy_loss.item())
        value_loss_meter.update(value_loss.item())
        total_loss_meter.update(total_loss.item())
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}: "
                       f"policy_loss={policy_loss.item():.4f}, "
                       f"value_loss={value_loss.item():.4f}, "
                       f"total_loss={total_loss.item():.4f}")
    
    return {
        'policy_loss': policy_loss_meter.avg,
        'value_loss': value_loss_meter.avg,
        'total_loss': total_loss_meter.avg
    }


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    rl_config = config['training']['reinforcement']
    
    # Setup logger
    logger = setup_logger('train_rl')
    logger.info(f"RL training configuration: {rl_config}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load self-play buffer
    logger.info(f"Loading self-play buffer from {args.buffer}")
    dataset = SelfPlayDataset(args.buffer)
    logger.info(f"Buffer size: {len(dataset)} positions")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=rl_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = create_model(config['model'])
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=rl_config['learning_rate']
    )
    
    # Load checkpoint
    checkpoint_info = load_checkpoint(args.model, model, optimizer)
    logger.info(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")
    
    # Train for specified iterations
    for iteration in range(rl_config['iterations']):
        logger.info(f"\nRL Iteration {iteration + 1}/{rl_config['iterations']}")
        
        # Train on current buffer
        metrics = train_iteration(
            model, dataloader, optimizer, device, logger,
            value_weight=rl_config['value_loss_weight']
        )
        log_metrics(logger, metrics, step=iteration + 1)
        
        # Save checkpoint
        checkpoint_path = Path(args.output_dir) / f"rl_iter_{iteration + 1:03d}.pt"
        save_checkpoint(
            model, optimizer, iteration + 1,
            metrics,
            checkpoint_path
        )
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # TODO: Generate new self-play games with updated model
        # TODO: Update buffer with new games
        # TODO: Evaluate model strength (arena matches)
    
    logger.info("\nRL training complete!")


if __name__ == '__main__':
    main()
