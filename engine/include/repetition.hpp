#pragma once

#include <vector>
#include <cstdint>

namespace chess {

class RepetitionTracker {
public:
    RepetitionTracker();

    void push(uint64_t intHash);
    void pop();
    void clear();

    bool is_repetition(uint64_t intHash, int intThreshold = 3) const;
    int count_occurrences(uint64_t intHash) const;

private:
    std::vector<uint64_t> vecHistory_;
};

} // namespace chess
