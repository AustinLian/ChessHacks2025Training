# Training Scripts

This directory contains scripts for training and evaluating the neural network.

## Scripts

### pgn_to_dataset.py
Convert PGN game files to training dataset.

```bash
python pgn_to_dataset.py \
    --input ../data/raw_pgn/*.pgn \
    --output ../data/processed/dataset_v1.npz \
    --max-games 100000 \
    --min-elo 2000
```

### train_supervised.py
Train model with supervised learning on PGN data.

```bash
python train_supervised.py \
    --config ../../config/training.yaml \
    --dataset ../data/processed/dataset_v1.npz \
    --output-dir ../../weights/checkpoints
```

### train_rl.py
Fine-tune model with reinforcement learning from self-play.

```bash
python train_rl.py \
    --config ../../config/training.yaml \
    --buffer ../../selfplay/buffers/rl_buffer/buffer.npz \
    --model ../../weights/checkpoints/best_model.pt \
    --output-dir ../../weights/checkpoints
```

### export_weights_npz.py
Export trained model to NPZ format for C++ engine.

```bash
python export_weights_npz.py \
    --model ../../weights/checkpoints/best_model.pt \
    --config ../../config/training.yaml \
    --output ../../weights/current/engine_weights.npz
```

### evaluate_model.py
Evaluate model accuracy on test set.

```bash
python evaluate_model.py \
    --model ../../weights/checkpoints/best_model.pt \
    --config ../../config/training.yaml \
    --test-data ../data/processed/test_set.npz \
    --top-k 3
```

## Pipeline

1. **Data Preparation**: Use `pgn_to_dataset.py` to convert PGN files
2. **Supervised Learning**: Train initial model with `train_supervised.py`
3. **Export for Engine**: Convert to C++ format with `export_weights_npz.py`
4. **Self-Play**: Generate games with the engine (see `selfplay/` directory)
5. **RL Training**: Fine-tune with `train_rl.py` on self-play data
6. **Evaluation**: Test with `evaluate_model.py`

Repeat steps 3-6 for iterative improvement.
