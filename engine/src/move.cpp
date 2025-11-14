#include "move.hpp"
#include <sstream>

namespace chess {

Move::Move(int from, int to, int promotion, int flags) {
    data_ = (from & 0x3F) | ((to & 0x3F) << 6) | ((promotion & 0x3) << 12) | ((flags & 0x3) << 14);
}

std::string Move::to_uci() const {
    if (data_ == 0) return "0000";
    
    int from_sq = from();
    int to_sq = to();
    
    // Convert square to algebraic notation (e.g., 0 -> a1, 63 -> h8)
    char from_file = 'a' + (from_sq % 8);
    char from_rank = '1' + (from_sq / 8);
    char to_file = 'a' + (to_sq % 8);
    char to_rank = '1' + (to_sq / 8);
    
    std::string result;
    result += from_file;
    result += from_rank;
    result += to_file;
    result += to_rank;
    
    // Add promotion piece if applicable
    if (is_promotion()) {
        int promo = promotion();
        switch (promo) {
            case 0: result += 'n'; break;  // Knight
            case 1: result += 'b'; break;  // Bishop
            case 2: result += 'r'; break;  // Rook
            case 3: result += 'q'; break;  // Queen
        }
    }
    
    return result;
}

Move Move::from_uci(const std::string& uci) {
    if (uci.length() < 4) {
        return Move();  // Invalid
    }
    
    // Parse from square
    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int from_sq = from_rank * 8 + from_file;
    
    // Parse to square
    int to_file = uci[2] - 'a';
    int to_rank = uci[3] - '1';
    int to_sq = to_rank * 8 + to_file;
    
    // Check for promotion
    int promotion = 0;
    int flags = 0;
    
    if (uci.length() == 5) {
        // Promotion move
        flags = 1;
        switch (uci[4]) {
            case 'n': promotion = 0; break;
            case 'b': promotion = 1; break;
            case 'r': promotion = 2; break;
            case 'q': promotion = 3; break;
        }
    }
    
    return Move(from_sq, to_sq, promotion, flags);
}

} // namespace chess
