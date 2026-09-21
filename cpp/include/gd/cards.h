// Cards, hands and the two orderings. See docs/RULES.md sections 3 and 4.
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace gd {

using CardId = uint8_t;

inline constexpr int kNumCardIds = 54;   // distinct ids
inline constexpr int kDeckSize = 108;    // two copies of each
inline constexpr int kNumRanks = 13;     // 2..A, index 0..12
inline constexpr int kNumSuits = 4;      // S=0, H=1, C=2, D=3
inline constexpr int kHearts = 1;
inline constexpr CardId kBJ = 52;        // black (small) joker
inline constexpr CardId kRJ = 53;        // red (big) joker
inline constexpr int kRankBJ = 13;       // joker "ranks" used by power()
inline constexpr int kRankRJ = 14;
inline constexpr int kNumPowers = 15;    // power values 0..14
inline constexpr int kHandSize = 27;

// Natural rank index. 0..12 for real ranks, 13 for BJ, 14 for RJ.
constexpr int rank_of(CardId c) { return c < 52 ? c / 4 : c - 39; }
// Suit index for real cards, -1 for jokers.
constexpr int suit_of(CardId c) { return c < 52 ? c % 4 : -1; }
constexpr bool is_joker(CardId c) { return c >= 52; }
constexpr CardId card_of(int rank, int suit) { return static_cast<CardId>(rank * 4 + suit); }
// The two heart cards of the round level are wild. Both share this id.
constexpr CardId wild_id(int level) { return static_cast<CardId>(level * 4 + kHearts); }

// Power order (RULES.md 4): the level rank is lifted above the ace.
constexpr int power(int rank, int level) {
  if (rank >= kRankBJ) return rank;
  if (rank == level) return 12;
  return rank < level ? rank : rank - 1;
}

// Sequence windows, natural order. Index 0 is always the ace-low window.
// Straights/straight flushes: 10 windows of 5 ranks. Tubes: 12 of 3. Plates: 13 of 2.
inline constexpr int kNumStraightWindows = 10;
inline constexpr int kNumTubeWindows = 12;
inline constexpr int kNumPlateWindows = 13;

// Rank bitmask (bit r set for rank r, 0..12) of each window.
uint16_t straight_window_mask(int idx);
uint16_t tube_window_mask(int idx);
uint16_t plate_window_mask(int idx);
// Ranks of a window, lowest natural rank first (ace first in window 0).
const std::array<int, 5>& straight_window(int idx);
const std::array<int, 3>& tube_window(int idx);
const std::array<int, 2>& plate_window(int idx);

// Text form: suit letter then rank character, "SB" and "HR" for the jokers.
int card_from_string(const std::string& s);          // -1 if unparsable
std::string card_to_string(CardId c);
std::vector<CardId> cards_from_string(const std::string& s);   // space separated
std::string cards_to_string(const std::vector<CardId>& cs);

// A multiset over the 54 card ids, at most two copies of each.
// Trivially copyable and 16 bytes so that states can be cloned cheaply.
struct Hand {
  uint64_t has1 = 0;   // bit c: at least one copy of card id c
  uint64_t has2 = 0;   // bit c: two copies

  constexpr int count(CardId c) const {
    const uint64_t b = uint64_t{1} << c;
    return ((has1 & b) ? 1 : 0) + ((has2 & b) ? 1 : 0);
  }
  constexpr void add(CardId c) {
    const uint64_t b = uint64_t{1} << c;
    if (has1 & b) has2 |= b; else has1 |= b;
  }
  constexpr void remove(CardId c) {
    const uint64_t b = uint64_t{1} << c;
    if (has2 & b) has2 &= ~b; else has1 &= ~b;
  }
  constexpr void add_n(CardId c, int n) { for (int i = 0; i < n; ++i) add(c); }
  constexpr bool contains(const Hand& o) const {
    // every card of o is present here with at least the same multiplicity
    return (o.has1 & ~has1) == 0 && (o.has2 & ~has2) == 0;
  }
  constexpr int size() const {
    return std::popcount(has1) + std::popcount(has2);
  }
  constexpr bool empty() const { return has1 == 0; }
  constexpr void clear() { has1 = 0; has2 = 0; }
  constexpr void add_all(const Hand& o) {
    // only valid while no card would exceed two copies
    const uint64_t both = has1 & o.has1;
    has2 |= o.has2 | both;
    has1 |= o.has1;
  }
  constexpr void remove_all(const Hand& o) {
    const uint64_t two = o.has2;             // drop both copies
    const uint64_t one = o.has1 & ~o.has2;   // drop one copy
    has1 &= ~two;
    has2 &= ~two;
    const uint64_t from_two = one & has2;    // two copies here, one leaves
    has2 &= ~from_two;
    has1 &= ~(one & ~from_two);
  }
  constexpr bool operator==(const Hand& o) const = default;

  std::vector<CardId> to_vector() const;
  static Hand from_vector(const std::vector<CardId>& cs);
  static Hand from_string(const std::string& s);
  std::string to_string() const;
};

// Derived views, recomputed once per decision rather than stored in Hand.
struct HandView {
  std::array<uint8_t, kNumPowers> rank_count{};   // by natural rank, 13=BJ, 14=RJ
  std::array<uint8_t, kNumCardIds> card_count{};
  std::array<uint16_t, kNumSuits> suit_rank_mask{};  // bit r: at least one card of rank r in that suit
  int wilds = 0;        // copies of wild_id(level) held
  int size = 0;         // total cards including wilds
  // rank_count excludes wild cards; card_count includes every card id as held.
};

HandView make_view(const Hand& h, int level);

}  // namespace gd
