#pragma once

#include <cstdint>
#include <string>

namespace chess {

// Move representation (16-bit)
// bits 0-5:  from square (0-63)
// bits 6-11: to square   (0-63)
// bits 12-13: promotion piece type (0=knight, 1=bishop, 2=rook, 3=queen; only valid for promotions)
// bits 14-15: special flags (0=normal, 1=promotion, 2=en-passant, 3=castling)

class Move {
public:
    Move();
    explicit Move(uint16_t intData);
    Move(int intFrom, int intTo);

    static Move make_promotion(int intFrom, int intTo, int intPromoCode);
    static Move make_en_passant(int intFrom, int intTo);
    static Move make_castling(int intFrom, int intTo);

    int from() const;
    int to() const;

    bool is_null() const;
    bool is_promotion() const;
    bool is_en_passant() const;
    bool is_castling() const;

    // 0..3: knight, bishop, rook, queen (only if promotion)
    int promotion_code() const;

    uint16_t data() const { return data_; }

    std::string to_uci() const;
    static Move from_uci(const std::string &strUCI);

    bool operator==(const Move &other) const { return data_ == other.data_; }
    bool operator!=(const Move &other) const { return data_ != other.data_; }

private:
    uint16_t data_;
};

} // namespace chess
