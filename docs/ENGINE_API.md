# Engine API Documentation

## C++ Interface Between Engine and Neural Network

This document describes the C++ API for neural network inference within the chess engine.

## Class: `NNInference`

Located in `engine/include/nn_infer.hpp`

### Purpose
Provides an interface for the chess engine to query the neural network for position evaluation.

### Key Methods

#### `bool load_weights(const std::string& weights_path)`
Load neural network weights from disk.

**Parameters:**
- `weights_path`: Path to NPZ file containing model weights

**Returns:**
- `true` if weights loaded successfully
- `false` on error

**Usage:**
```cpp
auto nn = std::make_shared<NNInference>();
if (!nn->load_weights("weights/current/engine_weights.npz")) {
    std::cerr << "Failed to load weights" << std::endl;
    return false;
}
```

#### `EvalResult evaluate(const Board& board, const std::vector<Move>& legal_moves)`
Evaluate a chess position and return policy + value.

**Parameters:**
- `board`: Current board position
- `legal_moves`: Vector of all legal moves in the position

**Returns:**
- `EvalResult` struct containing:
  - `std::vector<float> policy`: Probability for each legal move
  - `float value`: Position evaluation (-1 to +1)

**Usage:**
```cpp
auto legal_moves = MoveGen::generate_legal_moves(board);
auto result = nn->evaluate(board, legal_moves);

// result.policy[i] = probability of legal_moves[i]
// result.value = expected outcome from current player's perspective
```

**Notes:**
- Policy vector has same length as `legal_moves`
- Policy values sum to approximately 1.0 (softmax)
- Value is from perspective of side to move (positive = favorable)

#### `std::vector<EvalResult> evaluate_batch(...)`
Batch evaluation for multiple positions.

**Parameters:**
- `boards`: Vector of board positions
- `legal_moves`: Vector of legal move lists (one per board)

**Returns:**
- Vector of `EvalResult` (one per position)

**Usage:**
```cpp
std::vector<Board> positions = {...};
std::vector<std::vector<Move>> moves = {...};
auto results = nn->evaluate_batch(positions, moves);
```

## Board to Neural Network Input

The `NNInference` class internally converts `Board` objects to the 27-plane representation expected by the neural network.

### Conversion Process

1. **Piece Planes (0-11)**: One-hot encoding of piece positions
   - White: Pawn, Knight, Bishop, Rook, Queen, King (planes 0-5)
   - Black: Pawn, Knight, Bishop, Rook, Queen, King (planes 6-11)

2. **History Planes (12-19)**: Previous board states
   - Currently: Placeholder (zeros)
   - Future: Last 4 positions for repetition detection

3. **Metadata Planes (20-26)**:
   - Castling rights (4 planes)
   - En passant square (1 plane)
   - Side to move (1 plane)
   - Move count for 50-move rule (1 plane)

### Internal Implementation

```cpp
void NNInference::board_to_planes(const Board& board, float* planes) const {
    // planes: pointer to 27 * 8 * 8 floats
    // Fill in piece positions, castling, etc.
}
```

## Weight File Format

The NPZ weight file contains raw NumPy arrays for each layer.

### Expected Structure

```
engine_weights.npz/
├─ conv_input.weight          (128, 27, 3, 3)
├─ conv_input.bn.weight       (128,)
├─ conv_input.bn.bias         (128,)
├─ res_blocks.0.conv1.weight  (128, 128, 3, 3)
├─ res_blocks.0.bn1.weight    (128,)
└─ ... (additional layers)
```

### Loading Process

1. Open NPZ file using a library (e.g., cnpy)
2. Load each array by name
3. Copy into internal weight buffers
4. Verify shapes match expected architecture

## Integration with Search

The neural network is called from within the search function:

```cpp
float Search::alpha_beta(Board& board, int depth, float alpha, float beta, bool is_max) {
    if (depth == 0) {
        // Leaf node: evaluate with NN
        auto legal_moves = MoveGen::generate_legal_moves(board);
        auto eval = nn_->evaluate(board, legal_moves);
        return eval.value;
    }
    
    // Continue search...
}
```

### Using Policy for Move Ordering

The policy output can improve move ordering:

```cpp
std::vector<Move> Search::order_moves(const Board& board, const std::vector<Move>& moves) {
    // Get NN policy
    auto eval = nn_->evaluate(board, moves);
    
    // Sort moves by policy probability
    std::vector<std::pair<Move, float>> move_probs;
    for (size_t i = 0; i < moves.size(); ++i) {
        move_probs.emplace_back(moves[i], eval.policy[i]);
    }
    
    std::sort(move_probs.begin(), move_probs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<Move> sorted_moves;
    for (const auto& mp : move_probs) {
        sorted_moves.push_back(mp.first);
    }
    
    return sorted_moves;
}
```

## Performance Considerations

### Caching
Consider caching NN evaluations in the transposition table:
- Store position hash + evaluation
- Reuse if position is re-encountered

### Batch Inference (Future)
For parallel search (e.g., thread pool):
- Collect multiple positions to evaluate
- Call `evaluate_batch()` once
- Distribute results back to search threads

### Fallback Evaluation
If NN fails to load or errors occur:
```cpp
if (!nn_->load_weights(path)) {
    // Fall back to material count
    use_fallback_evaluation = true;
}
```

## Error Handling

All NN operations should handle errors gracefully:
- Failed weight loading → return false
- Invalid board state → return neutral evaluation
- Out of memory → fallback to simpler evaluation

## Thread Safety

Current implementation:
- Single-threaded inference
- No thread safety guarantees

Future multi-threading:
- Use mutex for NN access
- Or create separate `NNInference` instance per thread
- Or use thread-local storage

## Testing

Unit tests should verify:
1. Weight loading (valid and invalid files)
2. Board encoding (matches Python implementation)
3. Evaluation output (valid range, proper formatting)
4. Integration with search (doesn't crash, improves play)

See `engine/tests/search_tests.cpp` for examples.
