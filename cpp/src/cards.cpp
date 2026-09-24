// Window tables, text parsing/printing and Hand/HandView helpers.
// See docs/RULES.md sections 3 and 4.
#include "gd/cards.h"

#include <array>
#include <cstring>

namespace gd {

namespace {

// Sequence windows, natural order. Index 0 is always the ace-low window.
// Ace (rank 12) sits below the 2 in window 0 and never wraps otherwise.
constexpr std::array<std::array<int, 5>, kNumStraightWindows> kStraightWindows = {{
    {12, 0, 1, 2, 3},
    {0, 1, 2, 3, 4},
    {1, 2, 3, 4, 5},
    {2, 3, 4, 5, 6},
    {3, 4, 5, 6, 7},
    {4, 5, 6, 7, 8},
    {5, 6, 7, 8, 9},
    {6, 7, 8, 9, 10},
    {7, 8, 9, 10, 11},
    {8, 9, 10, 11, 12},
}};

constexpr std::array<std::array<int, 3>, kNumTubeWindows> kTubeWindows = {{
    {12, 0, 1},
    {0, 1, 2},
    {1, 2, 3},
    {2, 3, 4},
    {3, 4, 5},
    {4, 5, 6},
    {5, 6, 7},
    {6, 7, 8},
    {7, 8, 9},
    {8, 9, 10},
    {9, 10, 11},
    {10, 11, 12},
}};

constexpr std::array<std::array<int, 2>, kNumPlateWindows> kPlateWindows = {{
    {12, 0},
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 4},
    {4, 5},
    {5, 6},
    {6, 7},
    {7, 8},
    {8, 9},
    {9, 10},
    {10, 11},
    {11, 12},
}};

template <typename Window>
uint16_t mask_of(const Window& w) {
  uint16_t m = 0;
  for (int r : w) m |= static_cast<uint16_t>(1u << r);
  return m;
}

}  // namespace

uint16_t straight_window_mask(int idx) { return mask_of(kStraightWindows[idx]); }
uint16_t tube_window_mask(int idx) { return mask_of(kTubeWindows[idx]); }
uint16_t plate_window_mask(int idx) { return mask_of(kPlateWindows[idx]); }

const std::array<int, 5>& straight_window(int idx) { return kStraightWindows[idx]; }
const std::array<int, 3>& tube_window(int idx) { return kTubeWindows[idx]; }
const std::array<int, 2>& plate_window(int idx) { return kPlateWindows[idx]; }

namespace {
constexpr const char* kRankChars = "23456789TJQKA";
constexpr const char* kSuitChars = "SHCD";
}  // namespace

int card_from_string(const std::string& s) {
  if (s == "SB") return kBJ;
  if (s == "HR") return kRJ;
  if (s.size() != 2) return -1;
  int suit = -1;
  for (int i = 0; i < kNumSuits; ++i) {
    if (s[0] == kSuitChars[i]) { suit = i; break; }
  }
  if (suit < 0) return -1;
  const char* p = std::strchr(kRankChars, s[1]);
  if (p == nullptr) return -1;
  int rank = static_cast<int>(p - kRankChars);
  return static_cast<int>(card_of(rank, suit));
}

std::string card_to_string(CardId c) {
  if (c == kBJ) return "SB";
  if (c == kRJ) return "HR";
  std::string out(2, ' ');
  out[0] = kSuitChars[suit_of(c)];
  out[1] = kRankChars[rank_of(c)];
  return out;
}

std::vector<CardId> cards_from_string(const std::string& s) {
  std::vector<CardId> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && s[i] == ' ') ++i;
    size_t start = i;
    while (i < s.size() && s[i] != ' ') ++i;
    if (i > start) {
      int c = card_from_string(s.substr(start, i - start));
      if (c >= 0) out.push_back(static_cast<CardId>(c));
    }
  }
  return out;
}

std::string cards_to_string(const std::vector<CardId>& cs) {
  std::string out;
  for (size_t i = 0; i < cs.size(); ++i) {
    if (i) out += ' ';
    out += card_to_string(cs[i]);
  }
  return out;
}

std::vector<CardId> Hand::to_vector() const {
  std::vector<CardId> out;
  out.reserve(size());
  for (int c = 0; c < kNumCardIds; ++c) {
    int n = count(static_cast<CardId>(c));
    for (int i = 0; i < n; ++i) out.push_back(static_cast<CardId>(c));
  }
  return out;
}

Hand Hand::from_vector(const std::vector<CardId>& cs) {
  Hand h;
  for (CardId c : cs) h.add(c);
  return h;
}

Hand Hand::from_string(const std::string& s) { return from_vector(cards_from_string(s)); }

std::string Hand::to_string() const { return cards_to_string(to_vector()); }

HandView make_view(const Hand& h, int level) {
  // Walk occupied card ids instead of scanning all 54 possible ids.
  HandView v{};
  const CardId wid = wild_id(level);
  for (uint64_t b = h.has1; b; b &= b - 1) {
    const CardId c = static_cast<CardId>(std::countr_zero(b));
    v.card_count[c] = 1;
    if (c == wid) ++v.wilds;
    else ++v.rank_count[rank_of(c)];
    if (c < 52) v.suit_rank_mask[suit_of(c)] |= static_cast<uint16_t>(1u << rank_of(c));
  }
  for (uint64_t b = h.has2; b; b &= b - 1) {
    const CardId c = static_cast<CardId>(std::countr_zero(b));
    v.card_count[c] = 2;
    if (c == wid) ++v.wilds;
    else ++v.rank_count[rank_of(c)];
  }
  v.size = h.size();
  return v;
}

}  // namespace gd
