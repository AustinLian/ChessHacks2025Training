#pragma once

#include "board.hpp"
#include "move.hpp"
#include <vector>

namespace chess {

class MoveGen {
public:
    static std::vector<Move> generate_legal_moves(const Board &stBoard);
    static std::vector<Move> generate_captures(const Board &stBoard);
    static bool is_legal(const Board &stBoard, const Move &stMove);
    static bool is_in_check(const Board &stBoard, Color enmSide);

private:
    static void generate_pawn_moves(const Board &stBoard,
                                    std::vector<Move> &vecQuiet,
                                    std::vector<Move> &vecCaps);
    static void generate_knight_moves(const Board &stBoard,
                                      std::vector<Move> &vecQuiet,
                                      std::vector<Move> &vecCaps);
    static void generate_bishop_moves(const Board &stBoard,
                                      std::vector<Move> &vecQuiet,
                                      std::vector<Move> &vecCaps);
    static void generate_rook_moves(const Board &stBoard,
                                    std::vector<Move> &vecQuiet,
                                    std::vector<Move> &vecCaps);
    static void generate_queen_moves(const Board &stBoard,
                                     std::vector<Move> &vecQuiet,
                                     std::vector<Move> &vecCaps);
    static void generate_king_moves(const Board &stBoard,
                                    std::vector<Move> &vecQuiet,
                                    std::vector<Move> &vecCaps);
    static void generate_castling_moves(const Board &stBoard,
                                        std::vector<Move> &vecQuiet);

    static bool is_attacked(const Board &stBoard, int intSq, Color enmBy);

    // helper
    static int file_of(int intSq) { return intSq % 8; }
    static int rank_of(int intSq) { return intSq / 8; }
};

// Apply a move to a board (no undo; use copy in search)
void apply_move(Board &stBoard, const Move &stMove);

} // namespace chess
