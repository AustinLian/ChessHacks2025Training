#include "time_manager.hpp"

namespace chess {

TimeManager::TimeManager()
    : intBaseTimeMs_(0),
      intIncrementMs_(0),
      intMoveOverheadMs_(10),
      fltTimeBufferRatio_(0.1f) {}

void TimeManager::set_time_control(int64_t intBaseTimeMs, int64_t intIncrementMs) {
    intBaseTimeMs_  = intBaseTimeMs;
    intIncrementMs_ = intIncrementMs;
}

void TimeManager::set_move_overhead(int64_t intOverheadMs) {
    intMoveOverheadMs_ = intOverheadMs;
}

int64_t TimeManager::allocate_time(int /*intMovesPlayed*/, int64_t intTimeRemainingMs) const {
    int64_t intSafeTime = static_cast<int64_t>(
        intTimeRemainingMs * (1.0f - fltTimeBufferRatio_)
    );

    int intEstimatedMovesLeft = 30;
    if (intEstimatedMovesLeft <= 0) intEstimatedMovesLeft = 1;

    int64_t intPerMove = intSafeTime / intEstimatedMovesLeft
                       + intIncrementMs_
                       - intMoveOverheadMs_;
    if (intPerMove < 10) intPerMove = 10;
    return intPerMove;
}

void TimeManager::update_time_used(int64_t /*intTimeUsedMs*/) {
    // For now, no internal model; kept for future extensions.
}

} // namespace chess
