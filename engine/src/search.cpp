#include "search.hpp"
#include "movegen.hpp"
#include <algorithm>
#include <limits>
#include <chrono>

namespace chess {

Search::Search()
    : ptrTimeManager_(nullptr),
      ptrRepTracker_(nullptr),
      blnShouldStop_(false),
      intNodesSearched_(0) {}

void Search::stop() {
    blnShouldStop_ = true;
}

std::vector<Move> Search::order_moves(const Board &stBoard,
                                      const std::vector<Move> &vecMoves) {
    NNPolicyValue stNN = nnEvaluate(stBoard);

    struct ScoredMove {
        Move  stMove;
        float fltScore;
    };

    std::vector<ScoredMove> vecScored;
    vecScored.reserve(vecMoves.size());

    for (const Move &stMove : vecMoves) {
        int intFrom = stMove.from();
        int intTo   = stMove.to();
        int intIdx  = intFrom * 64 + intTo;
        float fltLogit = 0.0f;
        if (intIdx >= 0 &&
            intIdx < static_cast<int>(stNN.vecPolicy.size())) {
            fltLogit = stNN.vecPolicy[intIdx];
        }
        vecScored.push_back({stMove, fltLogit});
    }

    std::sort(vecScored.begin(), vecScored.end(),
              [](const ScoredMove &a, const ScoredMove &b) {
                  return a.fltScore > b.fltScore;
              });

    std::vector<Move> vecOrdered;
    vecOrdered.reserve(vecScored.size());
    for (const auto &sm : vecScored) {
        vecOrdered.push_back(sm.stMove);
    }
    return vecOrdered;
}

float Search::alpha_beta(Board &stBoard, int intDepth, float fltAlpha, float fltBeta) {
    if (blnShouldStop_) return 0.0f;

    intNodesSearched_++;

    if (intDepth <= 0) {
        NNPolicyValue stNN = nnEvaluate(stBoard);
        float fltVal = stNN.fltValue;
        if (stBoard.clrSideToMove == COLOR_BLACK) {
            fltVal = -fltVal;
        }
        return fltVal * 1000.0f;
    }

    auto vecMoves = MoveGen::generate_legal_moves(stBoard);
    if (vecMoves.empty()) {
        if (MoveGen::is_in_check(stBoard, stBoard.clrSideToMove)) {
            return -100000.0f + (float)(3 - intDepth);
        } else {
            return 0.0f;
        }
    }

    vecMoves = order_moves(stBoard, vecMoves);

    float fltBest = -std::numeric_limits<float>::infinity();

    for (const Move &stMove : vecMoves) {
        Board stChild = stBoard;
        apply_move(stChild, stMove);

        float fltScore = -alpha_beta(stChild, intDepth - 1, -fltBeta, -fltAlpha);
        if (fltScore > fltBest) {
            fltBest = fltScore;
        }
        if (fltScore > fltAlpha) {
            fltAlpha = fltScore;
        }
        if (fltAlpha >= fltBeta) {
            break;
        }
    }

    return fltBest;
}

SearchResult Search::search(Board &stBoard, int intMaxDepth, int64_t intTimeMs) {
    blnShouldStop_   = false;
    intNodesSearched_ = 0;

    auto tpStart = std::chrono::steady_clock::now();
    int64_t intTimeBudget = intTimeMs;
    if (ptrTimeManager_) {
        intTimeBudget = ptrTimeManager_->allocate_time(0, intTimeMs);
    }

    Move stBestMove;
    float fltBestScore = 0.0f;

    for (int intDepth = 1; intDepth <= intMaxDepth; ++intDepth) {
        auto tpNow = std::chrono::steady_clock::now();
        int64_t intElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(tpNow - tpStart).count();
        if (intElapsedMs >= intTimeBudget) {
            break;
        }

        auto vecMoves = MoveGen::generate_legal_moves(stBoard);
        if (vecMoves.empty()) {
            break;
        }

        vecMoves = order_moves(stBoard, vecMoves);

        float fltAlpha = -std::numeric_limits<float>::infinity();
        float fltBeta  =  std::numeric_limits<float>::infinity();

        Move stLocalBest;
        float fltLocalBest = -std::numeric_limits<float>::infinity();

        for (const Move &stMove : vecMoves) {
            Board stChild = stBoard;
            apply_move(stChild, stMove);

            float fltScore = -alpha_beta(stChild, intDepth - 1, -fltBeta, -fltAlpha);
            if (fltScore > fltLocalBest) {
                fltLocalBest = fltScore;
                stLocalBest  = stMove;
            }
            if (fltScore > fltAlpha) {
                fltAlpha = fltScore;
            }
            if (fltAlpha >= fltBeta) {
                break;
            }

            tpNow = std::chrono::steady_clock::now();
            intElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(tpNow - tpStart).count();
            if (intElapsedMs >= intTimeBudget) {
                break;
            }
        }

        if (!stLocalBest.is_null()) {
            stBestMove   = stLocalBest;
            fltBestScore = fltLocalBest;
        }

        tpNow = std::chrono::steady_clock::now();
        intElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(tpNow - tpStart).count();
        if (intElapsedMs >= intTimeBudget) {
            break;
        }
    }

    SearchResult stRes;
    stRes.stBestMove = stBestMove;
    stRes.fltScore   = fltBestScore;
    return stRes;
}

} // namespace chess
