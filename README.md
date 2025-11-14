# ChessHacks2025

Neural network-powered chess engine for ChessHacks2025 competition.

## Quick Start

### Build the Engine

```bash
# Build C++ engine
make engine

# Or build from engine directory
cd engine
make
```

### Run the Engine

```bash
# Run engine (UCI mode)
./engine/build/chesshacks_engine

# Or from engine directory
cd engine
make run
```

### Run Tests

```bash
# Run all tests
make tests

# Or from engine directory
cd engine
make tests
```

## Project Structure

```
chesshacks-engine/
├── engine/           # C++ chess engine
├── training/         # Python NN training
├── selfplay/         # Self-play generation
├── config/           # Configuration files
├── weights/          # Neural network weights
├── tools/            # Utility scripts
└── docs/             # Documentation
```

## Python Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python training/scripts/train_supervised.py \
    --dataset training/data/processed/dataset.npz \
    --output-dir weights/checkpoints
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design
- [Engine API](docs/ENGINE_API.md) - C++ API documentation
- [Training Pipeline](docs/TRAINING_PIPELINE.md) - How to train models
- [Competition Compliance](docs/CHESSHACKS_COMPLIANCE.md) - Rules compliance
- [Contributing](docs/CONTRIBUTING.md) - Contribution guidelines

## Requirements

- C++ compiler with C++17 support (GCC 7+, Clang 5+)
- Python 3.8+
- PyTorch 2.0+
- Make

## License

MIT License - see [LICENSE](LICENSE) file for details.