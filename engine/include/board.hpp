#pragma once
#include <cstdint>
#include <string>

namespace chess {

enum Color {
    COLOR_WHITE = 0,
    COLOR_BLACK = 1
};

enum PieceType {
    PIECE_PAWN,
    PIECE_KNIGHT,
    PIECE_BISHOP,
    PIECE_ROOK,
    PIECE_QUEEN,
    PIECE_KING,
    PIECE_NONE
};

enum CastlingRights : uint8_t {
    CASTLE_NONE        = 0,
    CASTLE_WHITE_KING  = 1 << 0,
    CASTLE_WHITE_QUEEN = 1 << 1,
    CASTLE_BLACK_KING  = 1 << 2,
    CASTLE_BLACK_QUEEN = 1 << 3
};

struct Board {
    PieceType arrPieceType[64];
    Color     arrPieceColor[64];

    Color   clrSideToMove;
    uint8_t bytCastlingRights;
    int8_t  intEnPassantSquare;   // -1 if none
    int     intHalfmoveClock;
    int     intFullmoveNumber;

    Board();

    void setFromFEN(const std::string &strFEN);
    std::string toFEN() const;
    std::string toString() const;
};

inline Color opposite_color(Color clr) {
    return (clr == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
}

} // namespace chess
