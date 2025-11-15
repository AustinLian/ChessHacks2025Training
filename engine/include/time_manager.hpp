#pragma once

#include <cstdint>

namespace chess {

class TimeManager {
public:
    TimeManager();

    void set_time_control(int64_t intBaseTimeMs, int64_t intIncrementMs);
    void set_move_overhead(int64_t intOverheadMs);

    int64_t allocate_time(int intMovesPlayed, int64_t intTimeRemainingMs) const;
    void update_time_used(int64_t intTimeUsedMs);

private:
    int64_t intBaseTimeMs_;
    int64_t intIncrementMs_;
    int64_t intMoveOverheadMs_;
    float   fltTimeBufferRatio_;
};

} // namespace chess
