# Architecture Overview

## System Design

ChessHacks Engine is a neural network-powered chess engine designed for competitive play. The system consists of three main components:

### 1. C++ Engine Core
- **Location**: `engine/`
- **Purpose**: Legal move generation, tree search, and gameplay
- **Key Features**:
  - Complete implementation of chess rules (castling, en passant, promotion)
  - Alpha-beta search with transposition table
  - Time management for different time controls
  - UCI/simple CLI protocol for interfacing

### 2. Neural Network Evaluation
- **Location**: `training/models/`
- **Purpose**: Position evaluation and move policy prediction
- **Architecture**: ResNet-based policy-value network
  - Input: 27-plane board representation (8×8 each)
  - Output: Policy head (4672 possible moves) + Value head (position score)
  
### 3. Training Pipeline
- **Location**: `training/` and `selfplay/`
- **Purpose**: Train and improve the neural network
- **Stages**:
  1. Supervised learning from PGN databases
  2. Self-play game generation
  3. Reinforcement learning from self-play

## Data Flow

```
PGN Games → Position Dataset → Supervised Training → Initial Model
                                                           ↓
                                                    Export Weights
                                                           ↓
                                                      C++ Engine
                                                           ↓
                                                      Self-Play
                                                           ↓
                                                    RL Training → Improved Model
                                                           ↓
                                                    (Repeat cycle)
```

## Component Interaction

### Training Phase
1. **Data Preparation**: PGN files → processed dataset (NPZ format)
2. **Supervised Learning**: Train on expert games
3. **Weight Export**: PyTorch model → NPZ arrays for C++
4. **Self-Play**: Engine plays against itself, collecting game data
5. **RL Training**: Improve model using self-play data
6. **Model Selection**: Arena matches to validate improvement

### Inference Phase
1. **Engine Startup**: Load weights from NPZ file
2. **Position Input**: Receive position (FEN or UCI moves)
3. **Board Encoding**: Convert to 27-plane representation
4. **NN Forward Pass**: Get policy and value predictions
5. **Search**: Use NN evaluation in alpha-beta search
6. **Move Selection**: Choose best move within time limit

## Board Representation

### Neural Network Input (27 planes, 8×8 each)
- Planes 0-5: White pieces (P, N, B, R, Q, K)
- Planes 6-11: Black pieces (P, N, B, R, Q, K)
- Planes 12-19: Position history (last 4 positions)
- Plane 20-23: Castling rights
- Plane 24: En passant square
- Plane 25: Side to move
- Plane 26: Move count (for 50-move rule)

### C++ Internal Representation
- Piece arrays: 64-element arrays for piece types and colors
- Bitboards could be added for faster move generation
- Zobrist hashing for transposition table

## Move Encoding

Moves are encoded as integers (0-4671):
- Regular moves: `from_square * 64 + to_square` (0-4095)
- Promotions: Special encoding for each promotion type (4096-4671)

## Search Strategy

1. **Iterative Deepening**: Start shallow, go deeper if time permits
2. **Alpha-Beta Pruning**: Cut branches that won't affect result
3. **Move Ordering**: 
   - TT move (from transposition table)
   - Captures (MVV-LVA)
   - Killer moves
   - History heuristic
4. **NN Evaluation**: Used at leaf nodes instead of hand-crafted evaluation

## Training Strategy

### Stage 1: Supervised Learning
- Dataset: High-quality PGN games (Elo > 2000)
- Loss: Cross-entropy (policy) + MSE (value)
- Goal: Learn basic chess knowledge

### Stage 2: Reinforcement Learning
- Self-play with current model
- Collect (position, search_policy, outcome) tuples
- Update model to match search policy and actual outcomes
- Goal: Improve beyond human level through self-improvement

### Stage 3: Continuous Improvement
- Generate new self-play games with improved model
- Maintain rolling buffer of recent games
- Periodic model evaluation via arena matches
- Only promote new model if it beats old model by threshold

## Performance Considerations

### C++ Engine
- Optimized move generation (legal moves only)
- Efficient board representation
- Fast transposition table lookups
- Minimized allocations in search

### Neural Network
- Batch size 1 for real-time play
- Optimized for low latency over throughput
- Consider quantization for deployment
- Possibility of ONNX export for portability

### Self-Play
- Parallel game generation
- Efficient storage (compressed NPZ)
- Asynchronous I/O for data collection

## Extensibility

The architecture supports:
- Different network architectures (edit `training/models/`)
- Custom evaluation functions (fallback when NN unavailable)
- Multiple time controls (via config files)
- Opening books for diverse self-play
- Model distillation for faster inference
