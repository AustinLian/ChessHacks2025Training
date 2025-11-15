#include "repetition.hpp"

namespace chess {

RepetitionTracker::RepetitionTracker() : vecHistory_() {}

void RepetitionTracker::push(uint64_t intHash) {
    vecHistory_.push_back(intHash);
}

void RepetitionTracker::pop() {
    if (!vecHistory_.empty()) {
        vecHistory_.pop_back();
    }
}

void RepetitionTracker::clear() {
    vecHistory_.clear();
}

int RepetitionTracker::count_occurrences(uint64_t intHash) const {
    int intCount = 0;
    for (uint64_t h : vecHistory_) {
        if (h == intHash) ++intCount;
    }
    return intCount;
}

bool RepetitionTracker::is_repetition(uint64_t intHash, int intThreshold) const {
    return count_occurrences(intHash) >= intThreshold;
}

} // namespace chess
