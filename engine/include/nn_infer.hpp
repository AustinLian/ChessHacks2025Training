#pragma once
#include "board.hpp"
#include <vector>
#include <string>

namespace chess {

struct NNPolicyValue {
    float              fltValue;   // [-1, 1], White POV
    std::vector<float> vecPolicy;  // logits for 64*64 move indices
};

bool blnLoadNNWeights(const std::string &strPath);
NNPolicyValue nnEvaluate(const Board &stBoard);

} // namespace chess
cd