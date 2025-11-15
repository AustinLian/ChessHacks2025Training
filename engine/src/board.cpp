#include "board.hpp"
#include <sstream>
#include <cctype>
#include <iostream>

namespace chess {

Board::Board() {
    for (int intSq = 0; intSq < 64; ++intSq) {
        arrPieceType[intSq]  = PIECE_NONE;
        arrPieceColor[intSq] = COLOR_WHITE;
    }
    clrSideToMove      = COLOR_WHITE;
    bytCastlingRights  = CASTLE_NONE;
    intEnPassantSquare = -1;
    intHalfmoveClock   = 0;
    intFullmoveNumber  = 1;
}

static int intFileOf(int intSq) { return intSq % 8; }
static int intRankOf(int intSq) { return intSq / 8; }
static int intSqIndex(int intFile, int intRank) { return intRank * 8 + intFile; }

void Board::setFromFEN(const std::string &strFEN) {
    if (strFEN == "startpos") {
        setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        return;
    }

    for (int intSq = 0; intSq < 64; ++intSq) {
        arrPieceType[intSq]  = PIECE_NONE;
        arrPieceColor[intSq] = COLOR_WHITE;
    }
    clrSideToMove      = COLOR_WHITE;
    bytCastlingRights  = CASTLE_NONE;
    intEnPassantSquare = -1;
    intHalfmoveClock   = 0;
    intFullmoveNumber  = 1;

    std::istringstream iss(strFEN);
    std::string strPlacement, strSide, strCastling, strEP;
    int intHalfmove = 0;
    int intFullmove = 1;

    iss >> strPlacement >> strSide >> strCastling >> strEP >> intHalfmove >> intFullmove;

    int intRank = 7;
    int intFile = 0;
    for (char ch : strPlacement) {
        if (ch == '/') {
            intRank--;
            intFile = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(ch))) {
            intFile += ch - '0';
            continue;
        }

        PieceType enmPiece = PIECE_NONE;
        Color     enmColor = std::isupper(static_cast<unsigned char>(ch)) ? COLOR_WHITE : COLOR_BLACK;
        char chLow = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

        switch (chLow) {
            case 'p': enmPiece = PIECE_PAWN;   break;
            case 'n': enmPiece = PIECE_KNIGHT; break;
            case 'b': enmPiece = PIECE_BISHOP; break;
            case 'r': enmPiece = PIECE_ROOK;   break;
            case 'q': enmPiece = PIECE_QUEEN;  break;
            case 'k': enmPiece = PIECE_KING;   break;
            default:  enmPiece = PIECE_NONE;   break;
        }

        if (intFile >= 0 && intFile < 8 && intRank >= 0 && intRank < 8) {
            int intSq = intSqIndex(intFile, intRank);
            arrPieceType[intSq]  = enmPiece;
            arrPieceColor[intSq] = enmColor;
        }
        intFile++;
    }

    if (!strSide.empty() && strSide[0] == 'b') {
        clrSideToMove = COLOR_BLACK;
    } else {
        clrSideToMove = COLOR_WHITE;
    }

    bytCastlingRights = CASTLE_NONE;
    if (strCastling != "-") {
        for (char c : strCastling) {
            if (c == 'K') bytCastlingRights |= CASTLE_WHITE_KING;
            if (c == 'Q') bytCastlingRights |= CASTLE_WHITE_QUEEN;
            if (c == 'k') bytCastlingRights |= CASTLE_BLACK_KING;
            if (c == 'q') bytCastlingRights |= CASTLE_BLACK_QUEEN;
        }
    }

    intEnPassantSquare = -1;
    if (strEP != "-" && strEP.size() == 2) {
        int intFileEP = strEP[0] - 'a';
        int intRankEP = strEP[1] - '1';
        if (intFileEP >= 0 && intFileEP < 8 && intRankEP >= 0 && intRankEP < 8) {
            intEnPassantSquare = intSqIndex(intFileEP, intRankEP);
        }
    }

    intHalfmoveClock = intHalfmove;
    intFullmoveNumber = intFullmove;
}

std::string Board::toFEN() const {
    std::ostringstream oss;

    for (int intRank = 7; intRank >= 0; --intRank) {
        int intEmpty = 0;
        for (int intFile = 0; intFile < 8; ++intFile) {
            int intSq = intSqIndex(intFile, intRank);
            PieceType enmPiece = arrPieceType[intSq];

            if (enmPiece == PIECE_NONE) {
                intEmpty++;
            } else {
                if (intEmpty > 0) {
                    oss << intEmpty;
                    intEmpty = 0;
                }
                Color enmClr = arrPieceColor[intSq];
                char ch = '1';
                switch (enmPiece) {
                    case PIECE_PAWN:   ch = 'p'; break;
                    case PIECE_KNIGHT: ch = 'n'; break;
                    case PIECE_BISHOP: ch = 'b'; break;
                    case PIECE_ROOK:   ch = 'r'; break;
                    case PIECE_QUEEN:  ch = 'q'; break;
                    case PIECE_KING:   ch = 'k'; break;
                    default:           ch = '1'; break;
                }
                if (enmClr == COLOR_WHITE) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
                oss << ch;
            }
        }
        if (intEmpty > 0) oss << intEmpty;
        if (intRank > 0) oss << '/';
    }

    oss << ' ' << (clrSideToMove == COLOR_WHITE ? 'w' : 'b') << ' ';

    if (bytCastlingRights == CASTLE_NONE) {
        oss << '-';
    } else {
        if (bytCastlingRights & CASTLE_WHITE_KING)  oss << 'K';
        if (bytCastlingRights & CASTLE_WHITE_QUEEN) oss << 'Q';
        if (bytCastlingRights & CASTLE_BLACK_KING)  oss << 'k';
        if (bytCastlingRights & CASTLE_BLACK_QUEEN) oss << 'q';
    }

    oss << ' ';

    if (intEnPassantSquare == -1) {
        oss << '-';
    } else {
        int intFileEP = intFileOf(intEnPassantSquare);
        int intRankEP = intRankOf(intEnPassantSquare);
        oss << static_cast<char>('a' + intFileEP)
            << static_cast<char>('1' + intRankEP);
    }

    oss << ' ' << intHalfmoveClock << ' ' << intFullmoveNumber;
    return oss.str();
}

std::string Board::toString() const {
    std::ostringstream oss;
    for (int intRank = 7; intRank >= 0; --intRank) {
        oss << (intRank + 1) << "  ";
        for (int intFile = 0; intFile < 8; ++intFile) {
            int intSq = intSqIndex(intFile, intRank);
            PieceType enmPiece = arrPieceType[intSq];
            char ch = '.';
            if (enmPiece != PIECE_NONE) {
                Color enmClr = arrPieceColor[intSq];
                switch (enmPiece) {
                    case PIECE_PAWN:   ch = 'p'; break;
                    case PIECE_KNIGHT: ch = 'n'; break;
                    case PIECE_BISHOP: ch = 'b'; break;
                    case PIECE_ROOK:   ch = 'r'; break;
                    case PIECE_QUEEN:  ch = 'q'; break;
                    case PIECE_KING:   ch = 'k'; break;
                    default:           ch = '.'; break;
                }
                if (enmClr == COLOR_WHITE) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            }
            oss << ch << ' ';
        }
        oss << '\n';
    }
    oss << "\n   a b c d e f g h\n";
    return oss.str();
}

} // namespace chess
