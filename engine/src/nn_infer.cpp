#include "nn_infer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

namespace chess {

namespace {

constexpr int kIntChannels      = 13;
constexpr int kIntBoardSquares  = 64;
constexpr int kIntInputSize     = kIntChannels * kIntBoardSquares; // 832
constexpr int kIntPolicySize    = 64 * 64;                         // 4096

static bool g_blnLoaded      = false;
static int  g_intHiddenSize  = 0;

static std::vector<float> g_vecW1;
static std::vector<float> g_vecB1;

static std::vector<float> g_vecWValue;
static float              g_fltBValue = 0.0f;

static std::vector<float> g_vecWPolicy;
static std::vector<float> g_vecBPolicy;

void encodeBoardToInput(const Board &stBoard, std::vector<float> &vecInput) {
    vecInput.assign(kIntInputSize, 0.0f);

    float fltSTM = (stBoard.clrSideToMove == COLOR_WHITE) ? 1.0f : -1.0f;

    for (int intSq = 0; intSq < kIntBoardSquares; ++intSq) {
        int intBase = intSq * kIntChannels;

        PieceType enmPiece = stBoard.arrPieceType[intSq];
        Color     enmColor = stBoard.arrPieceColor[intSq];

        if (enmPiece != PIECE_NONE) {
            int intChannel = -1;
            if (enmColor == COLOR_WHITE) {
                switch (enmPiece) {
                    case PIECE_PAWN:   intChannel = 0; break;
                    case PIECE_KNIGHT: intChannel = 1; break;
                    case PIECE_BISHOP: intChannel = 2; break;
                    case PIECE_ROOK:   intChannel = 3; break;
                    case PIECE_QUEEN:  intChannel = 4; break;
                    case PIECE_KING:   intChannel = 5; break;
                    default: break;
                }
            } else {
                switch (enmPiece) {
                    case PIECE_PAWN:   intChannel = 6;  break;
                    case PIECE_KNIGHT: intChannel = 7;  break;
                    case PIECE_BISHOP: intChannel = 8;  break;
                    case PIECE_ROOK:   intChannel = 9;  break;
                    case PIECE_QUEEN:  intChannel = 10; break;
                    case PIECE_KING:   intChannel = 11; break;
                    default: break;
                }
            }
            if (intChannel >= 0) {
                vecInput[intBase + intChannel] = 1.0f;
            }
        }

        vecInput[intBase + 12] = fltSTM;
    }
}

bool blnReadFloats(std::istream &ifs, std::vector<float> &vecOut, int intCount) {
    vecOut.resize(intCount);
    for (int i = 0; i < intCount; ++i) {
        if (!(ifs >> vecOut[i])) return false;
    }
    return true;
}

NNPolicyValue forwardNetwork(const std::vector<float> &vecInput) {
    assert(g_blnLoaded);
    assert(static_cast<int>(vecInput.size()) == kIntInputSize);

    NNPolicyValue stOut;
    stOut.vecPolicy.resize(kIntPolicySize);

    std::vector<float> vecHidden(g_intHiddenSize, 0.0f);

    for (int intH = 0; intH < g_intHiddenSize; ++intH) {
        float fltSum = g_vecB1[intH];
        int intRowOffset = intH * kIntInputSize;
        for (int intI = 0; intI < kIntInputSize; ++intI) {
            fltSum += g_vecW1[intRowOffset + intI] * vecInput[intI];
        }
        vecHidden[intH] = std::max(0.0f, fltSum);
    }

    float fltVal = g_fltBValue;
    for (int intH = 0; intH < g_intHiddenSize; ++intH) {
        fltVal += g_vecWValue[intH] * vecHidden[intH];
    }
    stOut.fltValue = std::tanh(fltVal);

    for (int intP = 0; intP < kIntPolicySize; ++intP) {
        float fltSum = g_vecBPolicy[intP];
        int intRowOffset = intP * g_intHiddenSize;
        for (int intH = 0; intH < g_intHiddenSize; ++intH) {
            fltSum += g_vecWPolicy[intRowOffset + intH] * vecHidden[intH];
        }
        stOut.vecPolicy[intP] = fltSum;
    }

    return stOut;
}

} // anonymous

bool blnLoadNNWeights(const std::string &strPath) {
    std::ifstream ifs(strPath);
    if (!ifs) {
        std::cerr << "blnLoadNNWeights: cannot open " << strPath << "\n";
        g_blnLoaded = false;
        return false;
    }

    int intFileInput = 0;
    int intFileHidden = 0;
    int intFilePolicy = 0;

    if (!(ifs >> intFileInput >> intFileHidden >> intFilePolicy)) {
        std::cerr << "blnLoadNNWeights: bad header in " << strPath << "\n";
        g_blnLoaded = false;
        return false;
    }

    if (intFileInput != kIntInputSize) {
        std::cerr << "blnLoadNNWeights: input size mismatch (" << intFileInput
                  << " vs " << kIntInputSize << ")\n";
        g_blnLoaded = false;
        return false;
    }
    if (intFilePolicy != kIntPolicySize) {
        std::cerr << "blnLoadNNWeights: policy size mismatch (" << intFilePolicy
                  << " vs " << kIntPolicySize << ")\n";
        g_blnLoaded = false;
        return false;
    }

    g_intHiddenSize = intFileHidden;

    int intW1Size      = g_intHiddenSize * kIntInputSize;
    int intB1Size      = g_intHiddenSize;
    int intWValueSize  = g_intHiddenSize;
    int intWPolicySize = kIntPolicySize * g_intHiddenSize;
    int intBPolicySize = kIntPolicySize;

    if (!blnReadFloats(ifs, g_vecW1, intW1Size)) {
        std::cerr << "blnLoadNNWeights: failed W1\n";
        g_blnLoaded = false;
        return false;
    }
    if (!blnReadFloats(ifs, g_vecB1, intB1Size)) {
        std::cerr << "blnLoadNNWeights: failed B1\n";
        g_blnLoaded = false;
        return false;
    }
    if (!blnReadFloats(ifs, g_vecWValue, intWValueSize)) {
        std::cerr << "blnLoadNNWeights: failed WValue\n";
        g_blnLoaded = false;
        return false;
    }
    if (!(ifs >> g_fltBValue)) {
        std::cerr << "blnLoadNNWeights: failed BValue\n";
        g_blnLoaded = false;
        return false;
    }
    if (!blnReadFloats(ifs, g_vecWPolicy, intWPolicySize)) {
        std::cerr << "blnLoadNNWeights: failed WPolicy\n";
        g_blnLoaded = false;
        return false;
    }
    if (!blnReadFloats(ifs, g_vecBPolicy, intBPolicySize)) {
        std::cerr << "blnLoadNNWeights: failed BPolicy\n";
        g_blnLoaded = false;
        return false;
    }

    g_blnLoaded = true;
    std::cerr << "blnLoadNNWeights: loaded hidden size " << g_intHiddenSize << "\n";
    return true;
}

NNPolicyValue nnEvaluate(const Board &stBoard) {
    if (!g_blnLoaded) {
        NNPolicyValue stOut;
        stOut.fltValue = 0.0f;
        stOut.vecPolicy.assign(kIntPolicySize, 0.0f);
        return stOut;
    }
    std::vector<float> vecInput;
    vecInput.reserve(kIntInputSize);
    encodeBoardToInput(stBoard, vecInput);
    return forwardNetwork(vecInput);
}

} // namespace chess
