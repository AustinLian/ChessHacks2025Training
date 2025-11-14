# Contributing to ChessHacks Engine

## Development Setup

### Prerequisites
- C++ compiler with C++17 support (GCC 7+, Clang 5+, or MSVC 2017+)
- Make build system
- Python 3.8+
- PyTorch 2.0+

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/AustinLian/ChessHacks2025.git
cd ChessHacks2025
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build the C++ engine:
```bash
cd engine
make
```

4. Run tests:
```bash
# C++ tests
cd engine
make tests

# Python tests
cd ../..
pytest training/tests
```

## Project Structure

- `engine/`: C++ chess engine
- `training/`: Python neural network training
- `selfplay/`: Self-play game generation
- `config/`: Configuration files
- `weights/`: Model weights
- `tools/`: Utility scripts
- `docs/`: Documentation

## Code Style

### C++ Code
- Follow standard C++ conventions
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes and types

### Python Code
- Follow PEP 8
- Use Black for formatting: `black training/ selfplay/ tools/`
- Maximum line length: 100 characters
- Use type hints where possible

## Making Changes

### GitHub Workflow

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make and test your changes
5. Push to your fork
6. Create a Pull Request on GitHub

## Testing

### C++ Tests
```bash
cd engine
make tests
```

### Python Tests
```bash
pytest
black training/ selfplay/ tools/
```

## Documentation

Update relevant documentation when making changes:
- `docs/ARCHITECTURE.md`: System design
- `docs/ENGINE_API.md`: C++ API
- `docs/TRAINING_PIPELINE.md`: Training process
- `README.md`: Quick start and overview

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
