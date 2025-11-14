# Training Pipeline

## Overview

The training pipeline transforms chess game data into a strong playing engine through three stages:
1. Supervised learning from expert games (PGN)
2. Self-play game generation
3. Reinforcement learning from self-play

## Stage 1: Supervised Learning from PGN

### Goal
Learn basic chess knowledge by imitating strong human players.

### Data Preparation

#### Step 1.1: Collect PGN Files
```bash
# Download high-quality games (Elo >= 2000)
# Place in training/data/raw_pgn/
```

#### Step 1.2: Convert to Dataset
```bash
python training/scripts/pgn_to_dataset.py \
    --input training/data/raw_pgn/*.pgn \
    --output training/data/processed/dataset_v1.npz \
    --max-games 100000 \
    --min-elo 2000
```

This creates:
- `planes`: Board representations (N, 27, 8, 8)
- `move_indices`: Played move indices (N,)
- `results`: Game outcomes (N,) as -1, 0, +1

#### Step 1.3: Train Model
```bash
python training/scripts/train_supervised.py \
    --config config/training.yaml \
    --dataset training/data/processed/dataset_v1.npz \
    --output-dir weights/checkpoints
```

Training loop:
1. Load batch of positions
2. Forward pass through network
3. Compute loss:
   - Policy loss: Cross-entropy between predicted policy and played move
   - Value loss: MSE between predicted value and game outcome
4. Backward pass and parameter update
5. Validate on held-out set

**Hyperparameters** (from `config/training.yaml`):
- Learning rate: 0.001 (with cosine schedule)
- Batch size: 512
- Epochs: 10
- Optimizer: Adam with weight decay

**Expected Results:**
- Move prediction accuracy: 40-50% (top-1)
- Value prediction error: < 0.3 (mean absolute error)

### Stage 1 Output
- Checkpoints in `weights/checkpoints/sup_epoch_*.pt`
- Best model in `weights/checkpoints/best_model.pt`

## Stage 2: Self-Play Generation

### Goal
Generate training data by having the engine play against itself.

### Export Weights for Engine

#### Step 2.1: Export to NPZ
```bash
python training/scripts/export_weights_npz.py \
    --model weights/checkpoints/best_model.pt \
    --config config/training.yaml \
    --output weights/current/engine_weights.npz
```

This converts PyTorch weights to NPZ format readable by C++ engine.

#### Step 2.2: Build Engine
```bash
cd engine
make
```

#### Step 2.3: Generate Self-Play Games
```bash
python selfplay/scripts/run_selfplay.py \
    --engine engine/build/chesshacks_engine \
    --weights weights/current/engine_weights.npz \
    --config config/selfplay.yaml \
    --num-games 1000 \
    --output selfplay/buffers/games \
    --workers 4
```

For each game:
1. Engine plays both sides
2. At each position, run search to get policy (visit counts)
3. Sample move from policy (with temperature)
4. Record: (position, search_policy, move)
5. At end of game, assign value to all positions based on outcome

**Configuration** (from `config/selfplay.yaml`):
- Nodes per move: 800
- Temperature schedule: 1.0 for first 30 moves, then 0.5
- Resign threshold: -0.95 for 5 consecutive moves

#### Step 2.4: Convert to Training Buffer
```bash
python selfplay/scripts/sample_to_rl_buffer.py \
    --games selfplay/buffers/games/games.json \
    --output selfplay/buffers/rl_buffer/buffer.npz \
    --buffer-size 500000
```

This creates:
- `planes`: Board representations
- `policies`: Search policies (from MCTS/engine search)
- `values`: Game outcomes

## Stage 3: Reinforcement Learning

### Goal
Improve the model by training on self-play data.

#### Step 3.1: RL Training
```bash
python training/scripts/train_rl.py \
    --config config/training.yaml \
    --buffer selfplay/buffers/rl_buffer/buffer.npz \
    --model weights/checkpoints/best_model.pt \
    --output-dir weights/checkpoints
```

Training differences from supervised:
- **Policy target**: Search policy (not just played move)
  - Loss: KL divergence with search policy
- **Value target**: Actual game outcome (same as supervised)
- **Buffer**: Rolling window of recent self-play games

**Hyperparameters:**
- Learning rate: 0.0005 (lower than supervised)
- Batch size: 1024
- Iterations: 100
- Buffer size: 500K positions

#### Step 3.2: Evaluate New Model
```bash
python selfplay/scripts/arena_matches.py \
    --engine engine/build/chesshacks_engine \
    --model1 weights/checkpoints/rl_iter_001.pt \
    --model2 weights/checkpoints/best_model.pt \
    --games 100 \
    --config config/selfplay.yaml
```

Play 100 games between new and old model:
- If new model wins by > 50 Elo, promote it as new best
- Otherwise, continue training or adjust hyperparameters

#### Step 3.3: Iteration
Repeat self-play → RL training → evaluation cycle:
1. Export new best model
2. Generate new self-play games
3. Train on new buffer
4. Evaluate against previous best
5. Promote if better

**Convergence:**
- Continue until Elo gain plateaus
- Typical: 10-20 iterations
- Monitor for overfitting to self-play

## Data Management

### Position Dataset v1 (Supervised)
- Source: PGN files
- Format: NPZ with (planes, move_indices, results)
- Size: ~1GB for 1M positions
- Split: 90% train, 10% validation

### Self-Play Buffer (RL)
- Source: Engine self-play
- Format: NPZ with (planes, policies, values)
- Size: Rolling buffer, max 500K positions
- Update: Add new games, evict oldest

### Archival
```bash
# Save buffer after each iteration
cp selfplay/buffers/rl_buffer/buffer.npz \
   selfplay/buffers/archives/buffer_iter_001.npz
```

## Monitoring Training

### Supervised Learning Metrics
- **Policy loss**: Should decrease steadily
- **Value loss**: Should decrease steadily
- **Move accuracy**: Top-1 should reach 40-50%
- **Value error**: MAE should reach < 0.3

### RL Training Metrics
- **Policy loss**: May increase initially (RL policy ≠ supervised)
- **Value loss**: Should decrease
- **Elo rating**: Arena matches show improvement

### TensorBoard
```bash
# During training
tensorboard --logdir=weights/checkpoints
```

Visualize:
- Loss curves
- Learning rate schedule
- Gradient norms
- Weight distributions

## Troubleshooting

### Supervised Training Issues

**Problem: High validation loss**
- Solution: More data or less complex model

**Problem: Slow convergence**
- Solution: Increase learning rate or batch size

### Self-Play Issues

**Problem: Games too short**
- Solution: Increase nodes per move or adjust resign threshold

**Problem: Repetitive play**
- Solution: Use opening book or increase temperature

### RL Training Issues

**Problem: Performance degrades**
- Solution: Overfitting to self-play; add more diversity

**Problem: No improvement**
- Solution: Increase buffer size or self-play games per iteration

## Best Practices

1. **Version Control**: Tag each model iteration
2. **Reproducibility**: Save configs with each checkpoint
3. **Evaluation**: Always validate with arena matches
4. **Data Quality**: Filter out low-quality self-play games
5. **Backup**: Keep previous best models before promoting new ones

## Advanced Techniques

### Knowledge Distillation
Train smaller model to imitate larger one:
```bash
python training/scripts/train_distill.py \
    --teacher weights/checkpoints/best_model.pt \
    --student tiny \
    --dataset training/data/processed/dataset_v1.npz
```

### Curriculum Learning
Start with easier positions, gradually increase difficulty.

### Auxiliary Tasks
Train on additional objectives:
- Piece count prediction
- Checkmate in N moves
- Tactical motif classification

These can improve value head accuracy.
