#include "board.hpp"
#include "movegen.hpp"
#include <iostream>
#include <vector>
#include <cstdint>

using namespace chess;

uint64_t intPerft(Board &stBoard, int intDepth) {
    if (intDepth == 0) return 1;

    auto vecMoves = MoveGen::generate_legal_moves(stBoard);
    uint64_t intNodes = 0;

    for (const Move &stMove : vecMoves) {
        Board stCopy = stBoard;
        apply_move(stCopy, stMove);
        intNodes += intPerft(stCopy, intDepth - 1);
    }
    return intNodes;
}

struct TestCase {
    const char *szName;
    const char *szFEN;
    int         intDepth;
    uint64_t    intExpected;
};

int main() {
    std::vector<TestCase> vecTests = {
        // Position 1: normal start position
        {
            "Startpos",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            5,
            4865609ULL
        },

        // Position 2: Kiwipete (castling, pins, etc.)
        {
            "Kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            4,
            4085603ULL
        },

        // Position 3: EP and rook races
        {
            "EP + rook race",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            4,
            43238ULL
        },

        // Position 4: promotions and crazy tactics
        {
            "Promotions",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            4,
            422333ULL
        }
    };

    for (const auto &t : vecTests) {
        Board stBoard;
        stBoard.setFromFEN(t.szFEN);

        uint64_t intNodes = intPerft(stBoard, t.intDepth);

        std::cout << t.szName << " depth " << t.intDepth
                  << " = " << intNodes
                  << " (expected " << t.intExpected << ")";
        if (intNodes == t.intExpected) {
            std::cout << "  OK\n";
        } else {
            std::cout << "  MISMATCH\n";
        }
    }

    return 0;
}
