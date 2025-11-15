#include "move.hpp"
#include <stdexcept>

namespace chess {

static const uint16_t MOVE_MASK_FROM   = 0x003F; // bits 0-5
static const uint16_t MOVE_MASK_TO     = 0x0FC0; // bits 6-11
static const uint16_t MOVE_MASK_PROMO  = 0x3000; // bits 12-13
static const uint16_t MOVE_MASK_FLAGS  = 0xC000; // bits 14-15

static const int MOVE_SHIFT_TO    = 6;
static const int MOVE_SHIFT_PROMO = 12;
static const int MOVE_SHIFT_FLAGS = 14;

Move::Move() : data_(0) {}

Move::Move(uint16_t intData) : data_(intData) {}

Move::Move(int intFrom, int intTo) {
    uint16_t intF = static_cast<uint16_t>(intFrom & 0x3F);
    uint16_t intT = static_cast<uint16_t>(intTo & 0x3F);
    data_ = intF | (intT << MOVE_SHIFT_TO);
}

Move Move::make_promotion(int intFrom, int intTo, int intPromoCode) {
    uint16_t intF = static_cast<uint16_t>(intFrom & 0x3F);
    uint16_t intT = static_cast<uint16_t>(intTo & 0x3F);
    uint16_t intP = static_cast<uint16_t>(intPromoCode & 0x3); // 0..3
    uint16_t intFlags = 1; // promotion flag
    uint16_t intData = intF | (intT << MOVE_SHIFT_TO)
                     | (intP << MOVE_SHIFT_PROMO)
                     | (intFlags << MOVE_SHIFT_FLAGS);
    return Move(intData);
}

Move Move::make_en_passant(int intFrom, int intTo) {
    uint16_t intF = static_cast<uint16_t>(intFrom & 0x3F);
    uint16_t intT = static_cast<uint16_t>(intTo & 0x3F);
    uint16_t intFlags = 2;
    uint16_t intData = intF | (intT << MOVE_SHIFT_TO)
                     | (intFlags << MOVE_SHIFT_FLAGS);
    return Move(intData);
}

Move Move::make_castling(int intFrom, int intTo) {
    uint16_t intF = static_cast<uint16_t>(intFrom & 0x3F);
    uint16_t intT = static_cast<uint16_t>(intTo & 0x3F);
    uint16_t intFlags = 3;
    uint16_t intData = intF | (intT << MOVE_SHIFT_TO)
                     | (intFlags << MOVE_SHIFT_FLAGS);
    return Move(intData);
}

int Move::from() const {
    return static_cast<int>(data_ & MOVE_MASK_FROM);
}

int Move::to() const {
    return static_cast<int>((data_ & MOVE_MASK_TO) >> MOVE_SHIFT_TO);
}

bool Move::is_null() const {
    return data_ == 0;
}

bool Move::is_promotion() const {
    return ((data_ & MOVE_MASK_FLAGS) >> MOVE_SHIFT_FLAGS) == 1;
}

bool Move::is_en_passant() const {
    return ((data_ & MOVE_MASK_FLAGS) >> MOVE_SHIFT_FLAGS) == 2;
}

bool Move::is_castling() const {
    return ((data_ & MOVE_MASK_FLAGS) >> MOVE_SHIFT_FLAGS) == 3;
}

int Move::promotion_code() const {
    if (!is_promotion()) return -1;
    return static_cast<int>((data_ & MOVE_MASK_PROMO) >> MOVE_SHIFT_PROMO);
}

static int intFileOf(int intSq) { return intSq % 8; }
static int intRankOf(int intSq) { return intSq / 8; }

std::string Move::to_uci() const {
    if (is_null()) return "0000";

    int intFromSq = from();
    int intToSq   = to();

    int intFromFile = intFileOf(intFromSq);
    int intFromRank = intRankOf(intFromSq);
    int intToFile   = intFileOf(intToSq);
    int intToRank   = intRankOf(intToSq);

    std::string str;
    str.reserve(5);
    str.push_back(static_cast<char>('a' + intFromFile));
    str.push_back(static_cast<char>('1' + intFromRank));
    str.push_back(static_cast<char>('a' + intToFile));
    str.push_back(static_cast<char>('1' + intToRank));

    if (is_promotion()) {
        int intPromo = promotion_code();
        char chPromo = 'q';
        switch (intPromo) {
            case 0: chPromo = 'n'; break;
            case 1: chPromo = 'b'; break;
            case 2: chPromo = 'r'; break;
            case 3: chPromo = 'q'; break;
            default: chPromo = 'q'; break;
        }
        str.push_back(chPromo);
    }

    return str;
}

Move Move::from_uci(const std::string &strUCI) {
    if (strUCI.size() < 4) {
        return Move();
    }

    int intFromFile = strUCI[0] - 'a';
    int intFromRank = strUCI[1] - '1';
    int intToFile   = strUCI[2] - 'a';
    int intToRank   = strUCI[3] - '1';

    if (intFromFile < 0 || intFromFile > 7 ||
        intFromRank < 0 || intFromRank > 7 ||
        intToFile   < 0 || intToFile   > 7 ||
        intToRank   < 0 || intToRank   > 7) {
        return Move();
    }

    int intFromSq = intFromRank * 8 + intFromFile;
    int intToSq   = intToRank   * 8 + intToFile;

    if (strUCI.size() == 5) {
        char chPromo = strUCI[4];
        int intPromoCode = 3; // default queen
        switch (chPromo) {
            case 'n': case 'N': intPromoCode = 0; break;
            case 'b': case 'B': intPromoCode = 1; break;
            case 'r': case 'R': intPromoCode = 2; break;
            case 'q': case 'Q': intPromoCode = 3; break;
            default:            intPromoCode = 3; break;
        }
        return make_promotion(intFromSq, intToSq, intPromoCode);
    }

    return Move(intFromSq, intToSq);
}

} // namespace chess
