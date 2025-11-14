#include <iostream>
#include <chrono>
#include "../../engine/include/board.hpp"
#include "../../engine/include/movegen.hpp"
#include "../../engine/include/search.hpp"
#include "../../engine/include/nn_infer.hpp"

using namespace chess;
using namespace std::chrono;

void benchmark_perft() {
    std::cout << "Benchmarking move generation (perft)...\n";
    
    Board board;
    
    auto start = steady_clock::now();
    uint64_t nodes = MoveGen::perft(board, 5);
    auto end = steady_clock::now();
    
    auto duration = duration_cast<milliseconds>(end - start).count();
    double nps = nodes / (duration / 1000.0);
    
    std::cout << "  Depth 5: " << nodes << " nodes\n";
    std::cout << "  Time: " << duration << " ms\n";
    std::cout << "  NPS: " << static_cast<uint64_t>(nps) << "\n";
}

void benchmark_search() {
    std::cout << "\nBenchmarking search...\n";
    
    Board board;
    auto nn = std::make_shared<NNInference>();
    Search search(nn);
    
    auto start = steady_clock::now();
    auto result = search.search(board, 10, 5000);
    auto end = steady_clock::now();
    
    auto duration = duration_cast<milliseconds>(end - start).count();
    
    std::cout << "  Depth: " << result.depth << "\n";
    std::cout << "  Nodes: " << result.nodes << "\n";
    std::cout << "  Time: " << duration << " ms\n";
    std::cout << "  NPS: " << (result.nodes * 1000 / duration) << "\n";
}

int main() {
    std::cout << "=== ChessHacks Engine Benchmark ===\n\n";
    
    // benchmark_perft();
    // benchmark_search();
    
    std::cout << "\nBenchmarks placeholder - implement when engine is complete\n";
    
    return 0;
}
