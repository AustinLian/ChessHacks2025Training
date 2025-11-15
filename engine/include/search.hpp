#pragma once

#include "board.hpp"
#include "move.hpp"
#include "time_manager.hpp"
#include "repetition.hpp"
#include "nn_infer.hpp"

#include <vector>
#include <memory>
#include <cstdint>

namespace chess {

struct SearchResult {
    Move  stBestMove;
    float fltScore;
};

class Search {
public:
    Search();

    void set_time_manager(std::shared_ptr<TimeManager> ptrTM) { ptrTimeManager_ = ptrTM; }
    void set_repetition_tracker(std::shared_ptr<RepetitionTracker> ptrRep) { ptrRepTracker_ = ptrRep; }

    SearchResult search(Board &stBoard, int intMaxDepth, int64_t intTimeMs);

    void stop();
    uint64_t nodes_searched() const { return intNodesSearched_; }

private:
    float alpha_beta(Board &stBoard, int intDepth, float fltAlpha, float fltBeta);
    std::vector<Move> order_moves(const Board &stBoard, const std::vector<Move> &vecMoves);

    std::shared_ptr<TimeManager>      ptrTimeManager_;
    std::shared_ptr<RepetitionTracker> ptrRepTracker_;

    bool      blnShouldStop_;
    uint64_t  intNodesSearched_;
};

} // namespace chess
