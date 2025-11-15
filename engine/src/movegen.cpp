#include "movegen.hpp"
#include <cstdlib>

namespace chess {

static int intSqIndex(int intFile, int intRank) {
    return intRank * 8 + intFile;
}

void MoveGen::generate_pawn_moves(const Board &stBoard,
                                  std::vector<Move> &vecQuiet,
                                  std::vector<Move> &vecCaps) {
    Color enmSide = stBoard.clrSideToMove;
    int intDir = (enmSide == COLOR_WHITE) ? 1 : -1;

    for (int intSq = 0; intSq < 64; ++intSq) {
        if (stBoard.arrPieceType[intSq] != PIECE_PAWN) continue;
        if (stBoard.arrPieceColor[intSq] != enmSide) continue;

        int intFile = file_of(intSq);
        int intRank = rank_of(intSq);
        int intNextRank = intRank + intDir;
        if (intNextRank < 0 || intNextRank > 7) continue;

        int intForwardSq = intSqIndex(intFile, intNextRank);

        // single push
        if (stBoard.arrPieceType[intForwardSq] == PIECE_NONE) {
            bool blnPromotionRank = (enmSide == COLOR_WHITE ? (intNextRank == 7) : (intNextRank == 0));
            if (blnPromotionRank) {
                for (int intPromo = 0; intPromo < 4; ++intPromo) {
                    vecQuiet.push_back(Move::make_promotion(intSq, intForwardSq, intPromo));
                }
            } else {
                vecQuiet.push_back(Move(intSq, intForwardSq));
            }

            // double push
            bool blnStartRank = (enmSide == COLOR_WHITE ? (intRank == 1) : (intRank == 6));
            if (blnStartRank) {
                int intNextRank2 = intRank + 2 * intDir;
                int intForwardSq2 = intSqIndex(intFile, intNextRank2);
                if (stBoard.arrPieceType[intForwardSq2] == PIECE_NONE) {
                    vecQuiet.push_back(Move(intSq, intForwardSq2));
                }
            }
        }

        // captures (including en passant)
        for (int intDF = -1; intDF <= 1; intDF += 2) {
            int intCapFile = intFile + intDF;
            int intCapRank = intRank + intDir;
            if (intCapFile < 0 || intCapFile > 7 ||
                intCapRank < 0 || intCapRank > 7) {
                continue;
            }
            int intCapSq = intSqIndex(intCapFile, intCapRank);
            PieceType enmTargetPiece = stBoard.arrPieceType[intCapSq];
            Color enmTargetColor = COLOR_WHITE;
            if (enmTargetPiece != PIECE_NONE) {
                enmTargetColor = stBoard.arrPieceColor[intCapSq];
            }

            bool blnNormalCap = (enmTargetPiece != PIECE_NONE && enmTargetColor != enmSide);
            bool blnEnPassant = (stBoard.intEnPassantSquare == intCapSq
                                 && enmTargetPiece == PIECE_NONE);

            if (blnNormalCap || blnEnPassant) {
                bool blnPromotionRank = (enmSide == COLOR_WHITE ? (intCapRank == 7) : (intCapRank == 0));
                if (blnPromotionRank) {
                    for (int intPromo = 0; intPromo < 4; ++intPromo) {
                        if (blnEnPassant) {
                            // (promotion + EP can’t actually happen, but keep branch consistent)
                            vecCaps.push_back(Move::make_en_passant(intSq, intCapSq));
                        } else {
                            vecCaps.push_back(Move::make_promotion(intSq, intCapSq, intPromo));
                        }
                    }
                } else {
                    if (blnEnPassant) {
                        vecCaps.push_back(Move::make_en_passant(intSq, intCapSq));
                    } else {
                        vecCaps.push_back(Move(intSq, intCapSq));
                    }
                }
            }
        }
    }
}

void MoveGen::generate_knight_moves(const Board &stBoard,
                                    std::vector<Move> &vecQuiet,
                                    std::vector<Move> &vecCaps) {
    Color enmSide = stBoard.clrSideToMove;
    static const int arrOffsets[8][2] = {
        { 1, 2}, { 2, 1}, { 2,-1}, { 1,-2},
        {-1,-2}, {-2,-1}, {-2, 1}, {-1, 2}
    };

    for (int intSq = 0; intSq < 64; ++intSq) {
        if (stBoard.arrPieceType[intSq] != PIECE_KNIGHT) continue;
        if (stBoard.arrPieceColor[intSq] != enmSide) continue;

        int intFile = file_of(intSq);
        int intRank = rank_of(intSq);

        for (int i = 0; i < 8; ++i) {
            int intNF = intFile + arrOffsets[i][0];
            int intNR = intRank + arrOffsets[i][1];
            if (intNF < 0 || intNF > 7 || intNR < 0 || intNR > 7) continue;
            int intToSq = intSqIndex(intNF, intNR);
            PieceType enmTargetPiece = stBoard.arrPieceType[intToSq];
            if (enmTargetPiece == PIECE_NONE) {
                vecQuiet.push_back(Move(intSq, intToSq));
            } else if (stBoard.arrPieceColor[intToSq] != enmSide) {
                vecCaps.push_back(Move(intSq, intToSq));
            }
        }
    }
}

void MoveGen::generate_bishop_moves(const Board &stBoard,
                                    std::vector<Move> &vecQuiet,
                                    std::vector<Move> &vecCaps) {
    Color enmSide = stBoard.clrSideToMove;
    static const int arrDirs[4][2] = {
        { 1, 1}, { 1,-1}, {-1, 1}, {-1,-1}
    };

    for (int intSq = 0; intSq < 64; ++intSq) {
        if (stBoard.arrPieceType[intSq] != PIECE_BISHOP) continue;
        if (stBoard.arrPieceColor[intSq] != enmSide) continue;

        int intFile = file_of(intSq);
        int intRank = rank_of(intSq);

        for (int i = 0; i < 4; ++i) {
            int intDF = arrDirs[i][0];
            int intDR = arrDirs[i][1];
            int intF = intFile + intDF;
            int intR = intRank + intDR;
            while (intF >= 0 && intF <= 7 && intR >= 0 && intR <= 7) {
                int intToSq = intSqIndex(intF, intR);
                PieceType enmTargetPiece = stBoard.arrPieceType[intToSq];
                if (enmTargetPiece == PIECE_NONE) {
                    vecQuiet.push_back(Move(intSq, intToSq));
                } else {
                    if (stBoard.arrPieceColor[intToSq] != enmSide) {
                        vecCaps.push_back(Move(intSq, intToSq));
                    }
                    break;
                }
                intF += intDF;
                intR += intDR;
            }
        }
    }
}

void MoveGen::generate_rook_moves(const Board &stBoard,
                                  std::vector<Move> &vecQuiet,
                                  std::vector<Move> &vecCaps) {
    Color enmSide = stBoard.clrSideToMove;
    static const int arrDirs[4][2] = {
        { 1, 0}, {-1, 0}, { 0, 1}, { 0,-1}
    };

    for (int intSq = 0; intSq < 64; ++intSq) {
        if (stBoard.arrPieceType[intSq] != PIECE_ROOK) continue;
        if (stBoard.arrPieceColor[intSq] != enmSide) continue;

        int intFile = file_of(intSq);
        int intRank = rank_of(intSq);

        for (int i = 0; i < 4; ++i) {
            int intDF = arrDirs[i][0];
            int intDR = arrDirs[i][1];
            int intF = intFile + intDF;
            int intR = intRank + intDR;
            while (intF >= 0 && intF <= 7 && intR >= 0 && intR <= 7) {
                int intToSq = intSqIndex(intF, intR);
                PieceType enmTargetPiece = stBoard.arrPieceType[intToSq];
                if (enmTargetPiece == PIECE_NONE) {
                    vecQuiet.push_back(Move(intSq, intToSq));
                } else {
                    if (stBoard.arrPieceColor[intToSq] != enmSide) {
                        vecCaps.push_back(Move(intSq, intToSq));
                    }
                    break;
                }
                intF += intDF;
                intR += intDR;
            }
        }
    }
}

void MoveGen::generate_queen_moves(const Board &stBoard,
                                   std::vector<Move> &vecQuiet,
                                   std::vector<Move> &vecCaps) {
    Color enmSide = stBoard.clrSideToMove;

    // 8 sliding directions (rook + bishop)
    static const int arrDirs[8][2] = {
        { 1, 0}, {-1, 0}, { 0, 1}, { 0,-1}, // rook-like
        { 1, 1}, { 1,-1}, {-1, 1}, {-1,-1}  // bishop-like
    };

    for (int intSq = 0; intSq < 64; ++intSq) {
        if (stBoard.arrPieceType[intSq] != PIECE_QUEEN) continue;
        if (stBoard.arrPieceColor[intSq] != enmSide) continue;

        int intFile = file_of(intSq);
        int intRank = rank_of(intSq);

        for (int i = 0; i < 8; ++i) {
            int intDF = arrDirs[i][0];
            int intDR = arrDirs[i][1];
            int intF  = intFile + intDF;
            int intR  = intRank + intDR;

            while (intF >= 0 && intF <= 7 && intR >= 0 && intR <= 7) {
                int intToSq = intSqIndex(intF, intR);
                PieceType enmTargetPiece = stBoard.arrPieceType[intToSq];

                if (enmTargetPiece == PIECE_NONE) {
                    vecQuiet.emplace_back(intSq, intToSq);
                } else {
                    if (stBoard.arrPieceColor[intToSq] != enmSide) {
                        vecCaps.emplace_back(intSq, intToSq);
                    }
                    break;
                }

                intF += intDF;
                intR += intDR;
            }
        }
    }
}

void MoveGen::generate_king_moves(const Board &stBoard,
                                  std::vector<Move> &vecQuiet,
                                  std::vector<Move> &vecCaps) {
    Color enmSide = stBoard.clrSideToMove;
    static const int arrOffsets[8][2] = {
        { 1, 0}, { 1, 1}, { 0, 1}, {-1, 1},
        {-1, 0}, {-1,-1}, { 0,-1}, { 1,-1}
    };

    for (int intSq = 0; intSq < 64; ++intSq) {
        if (stBoard.arrPieceType[intSq] != PIECE_KING) continue;
        if (stBoard.arrPieceColor[intSq] != enmSide) continue;

        int intFile = file_of(intSq);
        int intRank = rank_of(intSq);

        for (int i = 0; i < 8; ++i) {
            int intNF = intFile + arrOffsets[i][0];
            int intNR = intRank + arrOffsets[i][1];
            if (intNF < 0 || intNF > 7 || intNR < 0 || intNR > 7) continue;
            int intToSq = intSqIndex(intNF, intNR);
            PieceType enmTargetPiece = stBoard.arrPieceType[intToSq];
            if (enmTargetPiece == PIECE_NONE) {
                vecQuiet.push_back(Move(intSq, intToSq));
            } else if (stBoard.arrPieceColor[intToSq] != enmSide) {
                vecCaps.push_back(Move(intSq, intToSq));
            }
        }
    }
}

void MoveGen::generate_castling_moves(const Board &stBoard,
                                      std::vector<Move> &vecQuiet) {
    Color enmSide = stBoard.clrSideToMove;
    uint8_t bytCR = stBoard.bytCastlingRights;

    int intKingSq = -1;
    for (int i = 0; i < 64; ++i) {
        if (stBoard.arrPieceType[i] == PIECE_KING &&
            stBoard.arrPieceColor[i] == enmSide) {
            intKingSq = i;
            break;
        }
    }
    if (intKingSq == -1) return;

    bool blnInCheck = is_in_check(stBoard, enmSide);
    if (blnInCheck) return;

    if (enmSide == COLOR_WHITE) {
        // king side: e1 to g1
        if (bytCR & CASTLE_WHITE_KING) {
            int intF1 = intSqIndex(5, 0);
            int intG1 = intSqIndex(6, 0);
            if (stBoard.arrPieceType[intF1] == PIECE_NONE &&
                stBoard.arrPieceType[intG1] == PIECE_NONE &&
                !is_attacked(stBoard, intF1, COLOR_BLACK) &&
                !is_attacked(stBoard, intG1, COLOR_BLACK)) {
                vecQuiet.push_back(Move::make_castling(intSqIndex(4, 0), intG1));
            }
        }
        // queen side: e1 to c1
        if (bytCR & CASTLE_WHITE_QUEEN) {
            int intD1 = intSqIndex(3, 0);
            int intC1 = intSqIndex(2, 0);
            int intB1 = intSqIndex(1, 0);
            if (stBoard.arrPieceType[intD1] == PIECE_NONE &&
                stBoard.arrPieceType[intC1] == PIECE_NONE &&
                stBoard.arrPieceType[intB1] == PIECE_NONE &&
                !is_attacked(stBoard, intD1, COLOR_BLACK) &&
                !is_attacked(stBoard, intC1, COLOR_BLACK)) {
                vecQuiet.push_back(Move::make_castling(intSqIndex(4, 0), intC1));
            }
        }
    } else {
        // black king side: e8 to g8
        if (bytCR & CASTLE_BLACK_KING) {
            int intF8 = intSqIndex(5, 7);
            int intG8 = intSqIndex(6, 7);
            if (stBoard.arrPieceType[intF8] == PIECE_NONE &&
                stBoard.arrPieceType[intG8] == PIECE_NONE &&
                !is_attacked(stBoard, intF8, COLOR_WHITE) &&
                !is_attacked(stBoard, intG8, COLOR_WHITE)) {
                vecQuiet.push_back(Move::make_castling(intSqIndex(4, 7), intG8));
            }
        }
        // black queen side: e8 to c8
        if (bytCR & CASTLE_BLACK_QUEEN) {
            int intD8 = intSqIndex(3, 7);
            int intC8 = intSqIndex(2, 7);
            int intB8 = intSqIndex(1, 7);
            if (stBoard.arrPieceType[intD8] == PIECE_NONE &&
                stBoard.arrPieceType[intC8] == PIECE_NONE &&
                stBoard.arrPieceType[intB8] == PIECE_NONE &&
                !is_attacked(stBoard, intD8, COLOR_WHITE) &&
                !is_attacked(stBoard, intC8, COLOR_WHITE)) {
                vecQuiet.push_back(Move::make_castling(intSqIndex(4, 7), intC8));
            }
        }
    }
}

bool MoveGen::is_attacked(const Board &stBoard, int intSq, Color enmBy) {
    int intFile = file_of(intSq);
    int intRank = rank_of(intSq);

    // pawns
    if (enmBy == COLOR_WHITE) {
        int intR = intRank - 1;
        if (intR >= 0) {
            if (intFile - 1 >= 0) {
                int intSq2 = intSqIndex(intFile - 1, intR);
                if (stBoard.arrPieceType[intSq2] == PIECE_PAWN &&
                    stBoard.arrPieceColor[intSq2] == COLOR_WHITE) {
                    return true;
                }
            }
            if (intFile + 1 <= 7) {
                int intSq2 = intSqIndex(intFile + 1, intR);
                if (stBoard.arrPieceType[intSq2] == PIECE_PAWN &&
                    stBoard.arrPieceColor[intSq2] == COLOR_WHITE) {
                    return true;
                }
            }
        }
    } else {
        int intR = intRank + 1;
        if (intR <= 7) {
            if (intFile - 1 >= 0) {
                int intSq2 = intSqIndex(intFile - 1, intR);
                if (stBoard.arrPieceType[intSq2] == PIECE_PAWN &&
                    stBoard.arrPieceColor[intSq2] == COLOR_BLACK) {
                    return true;
                }
            }
            if (intFile + 1 <= 7) {
                int intSq2 = intSqIndex(intFile + 1, intR);
                if (stBoard.arrPieceType[intSq2] == PIECE_PAWN &&
                    stBoard.arrPieceColor[intSq2] == COLOR_BLACK) {
                    return true;
                }
            }
        }
    }

    // knights
    static const int arrKnight[8][2] = {
        { 1, 2}, { 2, 1}, { 2,-1}, { 1,-2},
        {-1,-2}, {-2,-1}, {-2, 1}, {-1, 2}
    };
    for (int i = 0; i < 8; ++i) {
        int intF = intFile + arrKnight[i][0];
        int intR = intRank + arrKnight[i][1];
        if (intF < 0 || intF > 7 || intR < 0 || intR > 7) continue;
        int intSq2 = intSqIndex(intF, intR);
        if (stBoard.arrPieceType[intSq2] == PIECE_KNIGHT &&
            stBoard.arrPieceColor[intSq2] == enmBy) {
            return true;
        }
    }

    // king
    static const int arrKing[8][2] = {
        { 1, 0}, { 1, 1}, { 0, 1}, {-1, 1},
        {-1, 0}, {-1,-1}, { 0,-1}, { 1,-1}
    };
    for (int i = 0; i < 8; ++i) {
        int intF = intFile + arrKing[i][0];
        int intR = intRank + arrKing[i][1];
        if (intF < 0 || intF > 7 || intR < 0 || intR > 7) continue;
        int intSq2 = intSqIndex(intF, intR);
        if (stBoard.arrPieceType[intSq2] == PIECE_KING &&
            stBoard.arrPieceColor[intSq2] == enmBy) {
            return true;
        }
    }

    // sliders: bishops / queens diagonals
    static const int arrBDirs[4][2] = {
        { 1, 1}, { 1,-1}, {-1, 1}, {-1,-1}
    };
    for (int i = 0; i < 4; ++i) {
        int intDF = arrBDirs[i][0];
        int intDR = arrBDirs[i][1];
        int intF = intFile + intDF;
        int intR = intRank + intDR;
        while (intF >= 0 && intF <= 7 && intR >= 0 && intR <= 7) {
            int intSq2 = intSqIndex(intF, intR);
            PieceType enmPiece = stBoard.arrPieceType[intSq2];
            if (enmPiece != PIECE_NONE) {
                if (stBoard.arrPieceColor[intSq2] == enmBy &&
                    (enmPiece == PIECE_BISHOP || enmPiece == PIECE_QUEEN)) {
                    return true;
                }
                break;
            }
            intF += intDF;
            intR += intDR;
        }
    }

    // sliders: rooks / queens orthogonal
    static const int arrRDirs[4][2] = {
        { 1, 0}, {-1, 0}, { 0, 1}, { 0,-1}
    };
    for (int i = 0; i < 4; ++i) {
        int intDF = arrRDirs[i][0];
        int intDR = arrRDirs[i][1];
        int intF = intFile + intDF;
        int intR = intRank + intDR;
        while (intF >= 0 && intF <= 7 && intR >= 0 && intR <= 7) {
            int intSq2 = intSqIndex(intF, intR);
            PieceType enmPiece = stBoard.arrPieceType[intSq2];
            if (enmPiece != PIECE_NONE) {
                if (stBoard.arrPieceColor[intSq2] == enmBy &&
                    (enmPiece == PIECE_ROOK || enmPiece == PIECE_QUEEN)) {
                    return true;
                }
                break;
            }
            intF += intDF;
            intR += intDR;
        }
    }

    return false;
}

bool MoveGen::is_in_check(const Board &stBoard, Color enmSide) {
    int intKingSq = -1;
    for (int intSq = 0; intSq < 64; ++intSq) {
        if (stBoard.arrPieceType[intSq] == PIECE_KING &&
            stBoard.arrPieceColor[intSq] == enmSide) {
            intKingSq = intSq;
            break;
        }
    }
    if (intKingSq == -1) return false;
    return is_attacked(stBoard, intKingSq, opposite_color(enmSide));
}

std::vector<Move> MoveGen::generate_legal_moves(const Board &stBoard) {
    std::vector<Move> vecQuiet;
    std::vector<Move> vecCaps;
    vecQuiet.reserve(32);
    vecCaps.reserve(32);

    generate_pawn_moves(stBoard, vecQuiet, vecCaps);
    generate_knight_moves(stBoard, vecQuiet, vecCaps);
    generate_bishop_moves(stBoard, vecQuiet, vecCaps);
    generate_rook_moves(stBoard, vecQuiet, vecCaps);
    generate_queen_moves(stBoard, vecQuiet, vecCaps);
    generate_king_moves(stBoard, vecQuiet, vecCaps);
    generate_castling_moves(stBoard, vecQuiet);

    std::vector<Move> vecResult;
    vecResult.reserve(vecQuiet.size() + vecCaps.size());

    // filter for king safety
    Color enmSide = stBoard.clrSideToMove;

    auto fnCheckMove = [&](const Move &stMove) {
        Board stCopy = stBoard;
        apply_move(stCopy, stMove);
        if (!is_in_check(stCopy, enmSide)) {
            vecResult.push_back(stMove);
        }
    };

    for (const Move &stMove : vecCaps)  fnCheckMove(stMove);
    for (const Move &stMove : vecQuiet) fnCheckMove(stMove);

    return vecResult;
}

std::vector<Move> MoveGen::generate_captures(const Board &stBoard) {
    std::vector<Move> vecQuiet;
    std::vector<Move> vecCaps;
    vecQuiet.reserve(32);
    vecCaps.reserve(32);

    generate_pawn_moves(stBoard, vecQuiet, vecCaps);
    generate_knight_moves(stBoard, vecQuiet, vecCaps);
    generate_bishop_moves(stBoard, vecQuiet, vecCaps);
    generate_rook_moves(stBoard, vecQuiet, vecCaps);
    generate_queen_moves(stBoard, vecQuiet, vecCaps);
    generate_king_moves(stBoard, vecQuiet, vecCaps);

    std::vector<Move> vecResult;
    vecResult.reserve(vecCaps.size());
    Color enmSide = stBoard.clrSideToMove;

    for (const Move &stMove : vecCaps) {
        Board stCopy = stBoard;
        apply_move(stCopy, stMove);
        if (!is_in_check(stCopy, enmSide)) {
            vecResult.push_back(stMove);
        }
    }
    return vecResult;
}

bool MoveGen::is_legal(const Board &stBoard, const Move &stMove) {
    auto vec = generate_legal_moves(stBoard);
    for (const Move &m : vec) {
        if (m == stMove) return true;
    }
    return false;
}

void apply_move(Board &stBoard, const Move &stMove) {
    int intFrom = stMove.from();
    int intTo   = stMove.to();

    PieceType enmPiece = stBoard.arrPieceType[intFrom];
    Color enmSide = stBoard.arrPieceColor[intFrom];

    PieceType enmCaptured = stBoard.arrPieceType[intTo];

    bool blnWasBlackToMove = (stBoard.clrSideToMove == COLOR_BLACK);

    // clear en-passant by default
    stBoard.intEnPassantSquare = -1;

    // en-passant capture
    if (stMove.is_en_passant() &&
        enmPiece == PIECE_PAWN &&
        enmCaptured == PIECE_NONE) {
        int intDir = (enmSide == COLOR_WHITE) ? 1 : -1;
        // captured pawn is one rank behind the destination
        int intCapSq = intTo - 8 * intDir;
        stBoard.arrPieceType[intCapSq]  = PIECE_NONE;
        stBoard.arrPieceColor[intCapSq] = COLOR_WHITE;
    }

    // move piece
    stBoard.arrPieceType[intTo]  = enmPiece;
    stBoard.arrPieceColor[intTo] = enmSide;
    stBoard.arrPieceType[intFrom]  = PIECE_NONE;
    stBoard.arrPieceColor[intFrom] = COLOR_WHITE;

    // promotion
    if (stMove.is_promotion() && enmPiece == PIECE_PAWN) {
        int intPromo = stMove.promotion_code();
        PieceType enmNew = PIECE_QUEEN;
        switch (intPromo) {
            case 0: enmNew = PIECE_KNIGHT; break;
            case 1: enmNew = PIECE_BISHOP; break;
            case 2: enmNew = PIECE_ROOK;   break;
            case 3: enmNew = PIECE_QUEEN;  break;
            default: enmNew = PIECE_QUEEN; break;
        }
        stBoard.arrPieceType[intTo] = enmNew;
    }

    // castling rook move
    if (enmPiece == PIECE_KING) {
        int intFileFrom = intFrom % 8;
        int intFileTo   = intTo   % 8;
        int intRank     = intFrom / 8;

        if (std::abs(intFileTo - intFileFrom) == 2) {
            if (intFileTo == 6) {
                // king side
                int intRookFrom = intRank * 8 + 7;
                int intRookTo   = intRank * 8 + 5;
                stBoard.arrPieceType[intRookTo]  = PIECE_ROOK;
                stBoard.arrPieceColor[intRookTo] = enmSide;
                stBoard.arrPieceType[intRookFrom]  = PIECE_NONE;
                stBoard.arrPieceColor[intRookFrom] = COLOR_WHITE;
            } else if (intFileTo == 2) {
                // queen side
                int intRookFrom = intRank * 8 + 0;
                int intRookTo   = intRank * 8 + 3;
                stBoard.arrPieceType[intRookTo]  = PIECE_ROOK;
                stBoard.arrPieceColor[intRookTo] = enmSide;
                stBoard.arrPieceType[intRookFrom]  = PIECE_NONE;
                stBoard.arrPieceColor[intRookFrom] = COLOR_WHITE;
            }
        }
    }

    // update castling rights on king or rook move / rook capture
    if (enmPiece == PIECE_KING) {
        if (enmSide == COLOR_WHITE) {
            stBoard.bytCastlingRights &= ~(CASTLE_WHITE_KING | CASTLE_WHITE_QUEEN);
        } else {
            stBoard.bytCastlingRights &= ~(CASTLE_BLACK_KING | CASTLE_BLACK_QUEEN);
        }
    }

    if (enmPiece == PIECE_ROOK) {
        int intRank = intFrom / 8;
        int intFile = intFrom % 8;
        if (enmSide == COLOR_WHITE) {
            if (intRank == 0 && intFile == 0) stBoard.bytCastlingRights &= ~CASTLE_WHITE_QUEEN;
            if (intRank == 0 && intFile == 7) stBoard.bytCastlingRights &= ~CASTLE_WHITE_KING;
        } else {
            if (intRank == 7 && intFile == 0) stBoard.bytCastlingRights &= ~CASTLE_BLACK_QUEEN;
            if (intRank == 7 && intFile == 7) stBoard.bytCastlingRights &= ~CASTLE_BLACK_KING;
        }
    }

    if (enmCaptured == PIECE_ROOK) {
        int intRank = intTo / 8;
        int intFile = intTo % 8;
        if (enmSide == COLOR_WHITE) {
            if (intRank == 7 && intFile == 0) stBoard.bytCastlingRights &= ~CASTLE_BLACK_QUEEN;
            if (intRank == 7 && intFile == 7) stBoard.bytCastlingRights &= ~CASTLE_BLACK_KING;
        } else {
            if (intRank == 0 && intFile == 0) stBoard.bytCastlingRights &= ~CASTLE_WHITE_QUEEN;
            if (intRank == 0 && intFile == 7) stBoard.bytCastlingRights &= ~CASTLE_WHITE_KING;
        }
    }

    // new en-passant square after double pawn push
    if (enmPiece == PIECE_PAWN && std::abs(intTo - intFrom) == 16) {
        stBoard.intEnPassantSquare = (intTo + intFrom) / 2;
    }

    // halfmove clock
    if (enmPiece == PIECE_PAWN || enmCaptured != PIECE_NONE) {
        stBoard.intHalfmoveClock = 0;
    } else {
        stBoard.intHalfmoveClock += 1;
    }

    // side to move
    stBoard.clrSideToMove = opposite_color(stBoard.clrSideToMove);

    // fullmove number
    if (blnWasBlackToMove) {
        stBoard.intFullmoveNumber += 1;
    }
}

} // namespace chess
