// interpret(), best_readings(), beats(), abstract ids and round bookkeeping.
// See docs/RULES.md sections 4, 5, 6 and 11.
#include "gd/rules.h"

#include <algorithm>
#include <array>

#include "gd/action.h"
#include "gd/cards.h"

namespace gd {

namespace {

// Nats occupy at most one distinct rank. Returns {false, -1} if they span
// more than one rank. When there are no nats at all (a combination made only
// of wild cards), the rank is forced to the round level (RULES.md 4.3).
struct SameRank { bool ok; int r; };

SameRank same_rank(const HandView& view, int n_nat, int level) {
  if (n_nat == 0) return {true, level};
  int found = -1;
  for (int i = 0; i < kNumPowers; ++i) {
    if (view.rank_count[i] > 0) {
      if (found != -1) return {false, -1};
      found = i;
    }
  }
  return {true, found};
}

void push_unique(std::vector<Reading>& out, Reading r) { out.push_back(r); }

}  // namespace

std::vector<Reading> interpret(const Hand& cards, int level, const RuleConfig& rules) {
  const int n = cards.size();
  std::vector<Reading> out;
  if (n == 0) {
    out.push_back(Reading{Type::Pass, 0, 0, -1});
    return out;
  }
  if (n > 10) return out;

  const HandView view = make_view(cards, level);
  const int w = view.wilds;
  const int n_nat = n - w;

  // ---- Single / Pair / Triple / Bomb: all nats share one rank ------------
  if (n == 1) {
    SameRank sr = same_rank(view, n_nat, level);
    if (sr.ok) push_unique(out, Reading{Type::Single, static_cast<int8_t>(power(sr.r, level)), 0, -1});
  }
  if (n == 2) {
    SameRank sr = same_rank(view, n_nat, level);
    if (sr.ok && (sr.r < kRankBJ || w == 0)) {
      push_unique(out, Reading{Type::Pair, static_cast<int8_t>(power(sr.r, level)), 0, -1});
    }
  }
  if (n == 3) {
    SameRank sr = same_rank(view, n_nat, level);
    if (sr.ok && sr.r < kRankBJ) {
      push_unique(out, Reading{Type::Triple, static_cast<int8_t>(power(sr.r, level)), 0, -1});
    }
  }
  if (n >= 4 && n <= 10) {
    SameRank sr = same_rank(view, n_nat, level);
    if (sr.ok && sr.r < kRankBJ) {
      push_unique(out, Reading{Type::Bomb, static_cast<int8_t>(power(sr.r, level)),
                                static_cast<int8_t>(n), -1});
    }
  }

  // ---- Joker bomb: exactly BJ BJ RJ RJ, never involves a wild ------------
  if (n == 4 && view.rank_count[kRankBJ] == 2 && view.rank_count[kRankRJ] == 2 && n_nat == 4) {
    push_unique(out, Reading{Type::JokerBomb, 0, 0, -1});
  }

  // ---- Full house: n == 5, a triple rank plus a distinct pair rank -------
  if (n == 5 && n_nat > 0) {
    for (int r = 0; r < kNumRanks; ++r) {
      const int deficit_r = 3 - view.rank_count[r];
      if (deficit_r < 0) continue;
      int best_p = -1;
      for (int p = kRankRJ; p >= 0; --p) {  // prefer the largest pair rank
        if (p == r) continue;
        if (p >= kRankBJ && !rules.full_house_joker_pair) continue;
        if (p >= kRankBJ) {
          if (view.rank_count[p] != 2) continue;  // wilds may never stand for a joker
        }
        const int deficit_p = 2 - view.rank_count[p];
        if (deficit_p < 0) continue;
        if (deficit_r + deficit_p != w) continue;
        const int other = n_nat - view.rank_count[r] - view.rank_count[p];
        if (other != 0) continue;
        best_p = p;
        break;
      }
      if (best_p >= 0) {
        push_unique(out, Reading{Type::FullHouse, static_cast<int8_t>(power(r, level)), 0,
                                  static_cast<int8_t>(power(best_p, level))});
      }
    }
  }

  // ---- Straight / straight flush: n == 5, five distinct natural ranks ----
  if (n == 5 && n_nat > 0 && view.rank_count[kRankBJ] == 0 && view.rank_count[kRankRJ] == 0) {
    bool ok = true;
    for (int r = 0; r < kNumRanks; ++r) {
      if (view.rank_count[r] > 1) { ok = false; break; }
    }
    if (ok) {
      uint16_t nat_mask = 0;
      for (int r = 0; r < kNumRanks; ++r) {
        if (view.rank_count[r] > 0) nat_mask |= static_cast<uint16_t>(1u << r);
      }
      bool flush = true;
      int common_suit = -1;
      const CardId wid = wild_id(level);
      for (CardId c : cards.to_vector()) {
        if (c == wid) continue;
        const int s = suit_of(c);
        if (common_suit == -1) common_suit = s;
        else if (s != common_suit) flush = false;
      }
      for (int widx = 0; widx < kNumStraightWindows; ++widx) {
        const uint16_t wmask = straight_window_mask(widx);
        if ((nat_mask & ~wmask) != 0) continue;
        if (5 - n_nat != w) continue;
        if (flush) push_unique(out, Reading{Type::StraightFlush, static_cast<int8_t>(widx), 0, -1});
        if (w > 0 || !flush) push_unique(out, Reading{Type::Straight, static_cast<int8_t>(widx), 0, -1});
      }
    }
  }

  // ---- Tube: n == 6, two cards per rank across a 3-rank window -----------
  if (n == 6 && n_nat > 0 && view.rank_count[kRankBJ] == 0 && view.rank_count[kRankRJ] == 0) {
    bool ok = true;
    for (int r = 0; r < kNumRanks; ++r) {
      if (view.rank_count[r] > 2) { ok = false; break; }
    }
    if (ok) {
      uint16_t nat_mask = 0;
      for (int r = 0; r < kNumRanks; ++r) {
        if (view.rank_count[r] > 0) nat_mask |= static_cast<uint16_t>(1u << r);
      }
      for (int widx = 0; widx < kNumTubeWindows; ++widx) {
        const uint16_t wmask = tube_window_mask(widx);
        if ((nat_mask & ~wmask) == 0) {
          push_unique(out, Reading{Type::Tube, static_cast<int8_t>(widx), 0, -1});
        }
      }
    }
  }

  // ---- Plate: n == 6, three cards per rank across a 2-rank window --------
  if (n == 6 && n_nat > 0 && view.rank_count[kRankBJ] == 0 && view.rank_count[kRankRJ] == 0) {
    bool ok = true;
    for (int r = 0; r < kNumRanks; ++r) {
      if (view.rank_count[r] > 3) { ok = false; break; }
    }
    if (ok) {
      uint16_t nat_mask = 0;
      for (int r = 0; r < kNumRanks; ++r) {
        if (view.rank_count[r] > 0) nat_mask |= static_cast<uint16_t>(1u << r);
      }
      for (int widx = 0; widx < kNumPlateWindows; ++widx) {
        const uint16_t wmask = plate_window_mask(widx);
        if ((nat_mask & ~wmask) == 0) {
          push_unique(out, Reading{Type::Plate, static_cast<int8_t>(widx), 0, -1});
        }
      }
    }
  }

  return out;
}

std::vector<Reading> best_readings(const Hand& cards, int level, const RuleConfig& rules) {
  const auto all = interpret(cards, level, rules);
  std::array<bool, static_cast<size_t>(Type::kNumTypes)> has{};
  std::array<Reading, static_cast<size_t>(Type::kNumTypes)> best{};
  for (const Reading& r : all) {
    const size_t idx = static_cast<size_t>(r.type);
    bool better;
    if (!has[idx]) {
      better = true;
    } else if (is_bomb_class(r.type)) {
      Action a; a.type = r.type; a.key = r.key; a.bomb_size = r.bomb_size;
      Action b; b.type = best[idx].type; b.key = best[idx].key; b.bomb_size = best[idx].bomb_size;
      better = strength(b) < strength(a);
    } else {
      better = r.key > best[idx].key;
    }
    if (better) { best[idx] = r; has[idx] = true; }
  }
  std::vector<Reading> out;
  for (size_t i = 0; i < has.size(); ++i) {
    if (has[i]) out.push_back(best[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// beats()

bool beats_reading(Type ct, int ckey, int cbomb, Type tt, int tkey, int tbomb) {
  if (tt == Type::Pass) return ct != Type::Pass;  // leading: any non-pass is legal
  if (ct == Type::Pass) return true;              // pass is always legal when following
  const bool cb = is_bomb_class(ct);
  const bool tb = is_bomb_class(tt);
  if (cb) {
    if (tb) {
      Action ca; ca.type = ct; ca.key = static_cast<int8_t>(ckey); ca.bomb_size = static_cast<int8_t>(cbomb);
      Action ta; ta.type = tt; ta.key = static_cast<int8_t>(tkey); ta.bomb_size = static_cast<int8_t>(tbomb);
      return strength(ta) < strength(ca);
    }
    return true;
  }
  if (tb) return false;
  return ct == tt && ckey > tkey;
}

bool beats(const Action& cand, const Action& top) {
  return beats_reading(cand.type, cand.key, cand.bomb_size, top.type, top.key, top.bomb_size);
}

// ---------------------------------------------------------------------------
// type_name / Action::to_string

const char* type_name(Type t) {
  switch (t) {
    case Type::Pass: return "Pass";
    case Type::Single: return "Single";
    case Type::Pair: return "Pair";
    case Type::Triple: return "Triple";
    case Type::FullHouse: return "FullHouse";
    case Type::Straight: return "Straight";
    case Type::Tube: return "Tube";
    case Type::Plate: return "Plate";
    case Type::Bomb: return "Bomb";
    case Type::StraightFlush: return "StraightFlush";
    case Type::JokerBomb: return "JokerBomb";
    case Type::Tribute: return "Tribute";
    case Type::BackTribute: return "BackTribute";
    default: return "?";
  }
}

std::string Action::to_string() const {
  std::string s = type_name(type);
  if (type == Type::Bomb) {
    s += " " + std::to_string(bomb_size) + "x" + std::to_string(key);
  } else if (type == Type::FullHouse) {
    s += " " + std::to_string(key) + "/" + std::to_string(fh_pair_rank);
  } else if (type != Type::Pass && type != Type::JokerBomb) {
    s += " " + std::to_string(key);
  }
  if (!cards.empty()) s += " [" + cards.to_string() + "]";
  return s;
}

// ---------------------------------------------------------------------------
// abstract ids (RULES.md 11.1, layout in action.h)

namespace {

// Straight-flush suit is erased from Reading, but a concrete Action's cards
// carry it: the suit shared by a strict majority of the five cards (at most
// two of them can be a wild, physically a heart card, so three real cards
// always agree on the flush suit).
int sf_suit_of(const Hand& cards) {
  std::array<int, kNumSuits> count{};
  for (CardId c : cards.to_vector()) {
    if (c < 52) ++count[suit_of(c)];
  }
  int best = 0;
  for (int s = 1; s < kNumSuits; ++s) {
    if (count[s] > count[best]) best = s;
  }
  return best;
}

// Full house pair slot: ranks 0..12 other than the triple's key keep their
// order (0..11), BJ pair is 12, RJ pair is 13.
int fh_pair_slot(int triple_key, int pair_key) {
  if (pair_key == kRankBJ) return 12;
  if (pair_key == kRankRJ) return 13;
  return pair_key < triple_key ? pair_key : pair_key - 1;
}

int fh_pair_from_slot(int triple_key, int slot) {
  if (slot == 12) return kRankBJ;
  if (slot == 13) return kRankRJ;
  return slot < triple_key ? slot : slot + 1;
}

}  // namespace

int abstract_id(const Action& a) {
  switch (a.type) {
    case Type::Pass: return kAbstractPass;
    case Type::Single: return kAbstractSingle + a.key;
    case Type::Pair: return kAbstractPair + a.key;
    case Type::Triple: return kAbstractTriple + a.key;
    case Type::FullHouse:
      return kAbstractFullHouse + a.key * 14 + fh_pair_slot(a.key, a.fh_pair_rank);
    case Type::Straight: return kAbstractStraight + a.key;
    case Type::Tube: return kAbstractTube + a.key;
    case Type::Plate: return kAbstractPlate + a.key;
    case Type::Bomb: return kAbstractBomb + a.key * 7 + (a.bomb_size - 4);
    case Type::StraightFlush: return kAbstractStraightFlush + a.key * 4 + sf_suit_of(a.cards);
    case Type::JokerBomb: return kAbstractJokerBomb;
    default: return -1;
  }
}

Action abstract_action(int id) {
  Action a;
  if (id == kAbstractPass) {
    a.type = Type::Pass;
  } else if (id < kAbstractPair) {
    a.type = Type::Single;
    a.key = static_cast<int8_t>(id - kAbstractSingle);
  } else if (id < kAbstractTriple) {
    a.type = Type::Pair;
    a.key = static_cast<int8_t>(id - kAbstractPair);
  } else if (id < kAbstractFullHouse) {
    a.type = Type::Triple;
    a.key = static_cast<int8_t>(id - kAbstractTriple);
  } else if (id < kAbstractStraight) {
    a.type = Type::FullHouse;
    const int off = id - kAbstractFullHouse;
    const int triple_key = off / 14;
    const int slot = off % 14;
    a.key = static_cast<int8_t>(triple_key);
    a.fh_pair_rank = static_cast<int8_t>(fh_pair_from_slot(triple_key, slot));
  } else if (id < kAbstractTube) {
    a.type = Type::Straight;
    a.key = static_cast<int8_t>(id - kAbstractStraight);
  } else if (id < kAbstractPlate) {
    a.type = Type::Tube;
    a.key = static_cast<int8_t>(id - kAbstractTube);
  } else if (id < kAbstractBomb) {
    a.type = Type::Plate;
    a.key = static_cast<int8_t>(id - kAbstractPlate);
  } else if (id < kAbstractStraightFlush) {
    a.type = Type::Bomb;
    const int off = id - kAbstractBomb;
    a.key = static_cast<int8_t>(off / 7);
    a.bomb_size = static_cast<int8_t>(off % 7 + 4);
  } else if (id < kAbstractJokerBomb) {
    a.type = Type::StraightFlush;
    const int off = id - kAbstractStraightFlush;
    a.key = static_cast<int8_t>(off / 4);
  } else {
    a.type = Type::JokerBomb;
  }
  return a;
}

// ---------------------------------------------------------------------------
// Round bookkeeping (RULES.md 8, mirrors gd_reference.level_gain/promote).

LevelGain level_gain(const std::array<int8_t, 4>& order) {
  const int banker = order[0];
  const int partner = (banker + 2) % 4;
  int idx = 0;
  for (int i = 0; i < 4; ++i) {
    if (order[i] == partner) { idx = i; break; }
  }
  static const int kGainByIdx[4] = {0, 3, 2, 1};
  return LevelGain{banker % 2, kGainByIdx[idx]};
}

int promote(int level, int gain) { return std::min(level + gain, 12); }

}  // namespace gd
