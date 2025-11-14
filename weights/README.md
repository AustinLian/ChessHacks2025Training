# Neural Network Weights

This directory contains trained neural network weights.

## Structure

- `current/`: Current production weights used by the engine
  - `engine_weights.npz`: Weight arrays in NPZ format for C++ inference
  - `engine_config.json`: Network architecture configuration
  - `VERSION`: Version identifier

- `checkpoints/`: Training checkpoints (PyTorch format)
  - `sup_epoch_XXX.pt`: Supervised training checkpoints
  - `rl_iter_XXX.pt`: RL training checkpoints
  - `best_model.pt`: Best model based on validation metrics

- `distilled/`: Distilled/compressed models
  - `tiny_net.pt`: Small model for fast inference
  - `tiny_net_engine.npz`: Exported for C++ engine

## Usage

### For Training (PyTorch)
Load checkpoints for continued training or evaluation:
```python
from training.utils import load_checkpoint
load_checkpoint('weights/checkpoints/best_model.pt', model, optimizer)
```

### For C++ Engine
The engine loads weights from `weights/current/engine_weights.npz`:
```bash
./engine/build/chesshacks_engine --weights weights/current/engine_weights.npz
```

### Exporting New Weights
After training, export to C++ format:
```bash
python training/scripts/export_weights_npz.py \
    --model weights/checkpoints/best_model.pt \
    --output weights/current/engine_weights.npz
```

## Weight Format

### PyTorch Checkpoints (.pt)
- Full model state dict
- Optimizer state
- Training metadata (epoch, metrics)

### Engine Weights (.npz)
- Raw weight arrays (NumPy)
- C-contiguous memory layout
- Separate arrays for each layer

## Notes

- Checkpoint files are excluded from git by default (large files)
- Only version-controlled: structure documentation and sample configs
- Current production weights should be backed up regularly
