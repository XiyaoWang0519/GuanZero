// Legal move generation. The contract is docs/MOVEGEN_SPEC.md.
#include "gd/movegen.h"

#include <algorithm>
#include <array>

#include "gd/rules.h"

namespace gd {
namespace {

// Inverse of power(): which natural rank carries this power at this level.
constexpr int rank_of_power(int p, int level) {
  if (p >= kRankBJ) return p;              // 13 = BJ, 14 = RJ
  if (p == 12) return level;
  return p < level ? p : p + 1;
}

// The distinct card ids that can serve one rank, split into the ids that must
// stay distinct and a pool of interchangeable ones (the suit_dedup reduction).
struct RankCands {
  std::array<CardId, 4> keep{};     // kept distinct, ascending
  std::array<uint8_t, 4> keep_n{};
  int n_keep = 0;
  std::array<CardId, 4> pool{};     // interchangeable, ascending
  std::array<uint8_t, 4> pool_n{};
  int n_pool = 0;
  int pool_total = 0;
  int total = 0;
};

struct Gen {
  const Hand& hand;
  const HandView& v;
  int level;
  CardId wid;
  const ActionConfig& acfg;
  const RuleConfig& rules;
  const Action& top;
  bool leading;
  std::vector<Action>& out;
  size_t first = 0;                             // where our output starts
  std::array<uint16_t, kNumSuits> nat_mask{};   // natural cards, wild excluded
  std::array<uint16_t, kNumSuits> sf_mask{};

  bool want(Type t, int key, int bomb_size) const {
    return beats_reading(t, key, bomb_size, top.type, top.key, top.bomb_size);
  }

  // `suit` restricts to one suit (straight flushes); -1 means any suit.
  RankCands candidates(int rank, int suit) const {
    RankCands c;
    if (rank >= kRankBJ) {
      const CardId id = rank == kRankBJ ? kBJ : kRJ;
      if (v.card_count[id]) {
        c.keep[c.n_keep] = id;
        c.keep_n[c.n_keep++] = v.card_count[id];
        c.total = v.card_count[id];
      }
      return c;
    }
    const bool dedup = acfg.suit_dedup;
    for (int s = 0; s < kNumSuits; ++s) {
      if (suit >= 0 && s != suit) continue;
      const CardId id = card_of(rank, s);
      if (id == wid) continue;                  // wild cards are never naturals
      const int n = v.card_count[id];
      if (n == 0) continue;
      c.total += n;
      const bool relevant = (sf_mask[s] >> rank) & 1u;
      if (!dedup || relevant) {
        c.keep[c.n_keep] = id;
        c.keep_n[c.n_keep++] = static_cast<uint8_t>(n);
      } else {
        c.pool[c.n_pool] = id;
        c.pool_n[c.n_pool++] = static_cast<uint8_t>(n);
        c.pool_total += n;
      }
    }
    return c;
  }

  // Adds the `k` lexicographically smallest cards of the pool to `acc`.
  static void take_pool(const RankCands& c, int k, Hand& acc) {
    for (int i = 0; i < c.n_pool && k > 0; ++i) {
      const int take = std::min<int>(k, c.pool_n[i]);
      acc.add_n(c.pool[i], take);
      k -= take;
    }
  }
  static void drop_pool(const RankCands& c, int k, Hand& acc) {
    for (int i = 0; i < c.n_pool && k > 0; ++i) {
      const int take = std::min<int>(k, c.pool_n[i]);
      for (int j = 0; j < take; ++j) acc.remove(c.pool[i]);
      k -= take;
    }
  }

  // Every multiset of exactly `take` cards out of this rank's candidates.
  template <typename F>
  void each_choice(const RankCands& c, int take, Hand& acc, F&& fn) const {
    std::array<int, 4> pick{};
    const int nk = c.n_keep;
    // Odometer over the kept ids, remainder comes from the pool.
    while (true) {
      int used = 0;
      for (int i = 0; i < nk; ++i) used += pick[i];
      const int rest = take - used;
      if (rest >= 0 && rest <= c.pool_total) {
        for (int i = 0; i < nk; ++i) acc.add_n(c.keep[i], pick[i]);
        take_pool(c, rest, acc);
        fn();
        drop_pool(c, rest, acc);
        for (int i = 0; i < nk; ++i)
          for (int j = 0; j < pick[i]; ++j) acc.remove(c.keep[i]);
      }
      int i = 0;
      for (; i < nk; ++i) {
        if (pick[i] < c.keep_n[i]) { ++pick[i]; break; }
        pick[i] = 0;
      }
      if (i == nk) break;
    }
  }

  struct Req { int rank; int need; };

  // Walks the required ranks, filling shortfalls with wild cards.
  template <typename F>
  void expand(const Req* reqs, int nreq, int idx, int wilds_left, int wilds_used,
              int suit, Hand& acc, F&& emit) {
    if (idx == nreq) {
      Hand cards = acc;
      cards.add_n(wid, wilds_used);
      emit(cards, wilds_used);
      return;
    }
    const Req r = reqs[idx];
    const RankCands c = candidates(r.rank, suit);
    const int hi = std::min(r.need, c.total);
    // A joker slot can never be filled by a wild card.
    const int floor = (r.rank >= kRankBJ) ? r.need
                                          : std::max(0, r.need - wilds_left);
    const int lo = (acfg.wild_usage == WildUsage::Minimal) ? hi : floor;
    for (int take = hi; take >= lo; --take) {
      if (take < floor) break;
      const int spend = r.need - take;
      if (spend > wilds_left) continue;
      each_choice(c, take, acc, [&] {
        expand(reqs, nreq, idx + 1, wilds_left - spend, wilds_used + spend, suit,
               acc, emit);
      });
    }
  }

  void add(Type t, int key, int bomb_size, int fh_pair, const Hand& cards, int wilds) {
    // A combination made only of wild cards is read at the level rank, never at
    // a rank the wild cards stand in for (RULES.md 4.3). Only singles and pairs
    // can reach this, since no hand holds more than two wild cards.
    if (cards.size() == wilds && key != 12) return;
    Action a;
    a.type = t;
    a.key = static_cast<int8_t>(key);
    a.bomb_size = static_cast<int8_t>(bomb_size);
    a.wilds = static_cast<int8_t>(wilds);
    a.fh_pair_rank = static_cast<int8_t>(fh_pair);
    a.cards = cards;
    out.push_back(a);
  }

  // ---- per type ----------------------------------------------------------

  void singles() {
    for (int p = 0; p < kNumPowers; ++p) {
      if (!want(Type::Single, p, 0)) continue;
      const Req req{rank_of_power(p, level), 1};
      Hand acc;
      expand(&req, 1, 0, v.wilds, 0, -1, acc,
             [&](const Hand& cards, int w) { add(Type::Single, p, 0, -1, cards, w); });
    }
  }

  void same_rank(Type t, int size) {
    const int max_p = (t == Type::Pair) ? kNumPowers : 13;
    for (int p = 0; p < max_p; ++p) {
      if (!want(t, p, t == Type::Bomb ? size : 0)) continue;
      const int rank = rank_of_power(p, level);
      if (t != Type::Pair && rank >= kRankBJ) continue;   // jokers make no triple or bomb
      const Req req{rank, size};
      Hand acc;
      expand(&req, 1, 0, v.wilds, 0, -1, acc, [&](const Hand& cards, int w) {
        add(t, p, t == Type::Bomb ? size : 0, -1, cards, w);
      });
    }
  }

  void joker_bomb() {
    if (!want(Type::JokerBomb, 0, 0)) return;
    if (v.card_count[kBJ] < 2 || v.card_count[kRJ] < 2) return;
    Hand cards;
    cards.add_n(kBJ, 2);
    cards.add_n(kRJ, 2);
    add(Type::JokerBomb, 0, 0, -1, cards, 0);
  }

  void full_houses() {
    for (int tp = 0; tp < 13; ++tp) {
      if (!want(Type::FullHouse, tp, 0)) continue;
      const int rt = rank_of_power(tp, level);
      for (int pp = 0; pp < kNumPowers; ++pp) {
        if (pp == tp) continue;
        const int rp = rank_of_power(pp, level);
        if (rp == rt) continue;
        if (rp >= kRankBJ && !rules.full_house_joker_pair) continue;
        const Req reqs[2] = {{rt, 3}, {rp, 2}};
        Hand acc;
        expand(reqs, 2, 0, v.wilds, 0, -1, acc, [&](const Hand& cards, int w) {
          add(Type::FullHouse, tp, 0, pp, cards, w);
        });
      }
    }
  }

  // Straights and straight flushes share their card multisets, so they are
  // generated together and classified afterwards (RULES.md 5 and the oracle).
  void straights() {
    const bool need_straight = want(Type::Straight, 0, 0) ||
                               (!leading && top.type == Type::Straight);
    bool need_sf = false;
    for (int w = 0; w < kNumStraightWindows; ++w)
      need_sf = need_sf || want(Type::StraightFlush, w, 0);
    if (!need_straight && !need_sf) return;

    for (int w = 0; w < kNumStraightWindows; ++w) {
      const auto& ranks = straight_window(w);
      const bool take_straight = want(Type::Straight, w, 0);
      const bool take_sf = want(Type::StraightFlush, w, 0);
      if (!take_straight && !take_sf) continue;
      Req reqs[5];
      for (int i = 0; i < 5; ++i) reqs[i] = Req{ranks[i], 1};

      if (take_straight) {
        Hand acc;
        expand(reqs, 5, 0, v.wilds, 0, -1, acc, [&](const Hand& cards, int wl) {
          // One suit across every natural card makes it a straight flush; with
          // a wild card in play it reads as either.
          int suit = -2;
          bool one_suit = true;
          for (int c = 0; c < 52 && one_suit; ++c) {
            if (c == wid || !cards.count(static_cast<CardId>(c))) continue;
            if (suit == -2) suit = suit_of(static_cast<CardId>(c));
            else if (suit != suit_of(static_cast<CardId>(c))) one_suit = false;
          }
          if (!one_suit || wl > 0) add(Type::Straight, w, 0, -1, cards, wl);
          if (one_suit && take_sf) add(Type::StraightFlush, w, 0, -1, cards, wl);
        });
      }
      // Straight flushes need their own pass, restricted to one suit. Picking
      // naturals suit-blind would let the minimal-wild rule take a card of the
      // wrong suit and lose the flush entirely.
      if (take_sf) {
        for (int s = 0; s < kNumSuits; ++s) {
          Hand acc;
          expand(reqs, 5, 0, v.wilds, 0, s, acc, [&](const Hand& cards, int wl) {
            add(Type::StraightFlush, w, 0, -1, cards, wl);
          });
        }
      }
    }
  }

  void sequences(Type t, int per_rank) {
    const int windows = t == Type::Tube ? kNumTubeWindows : kNumPlateWindows;
    for (int w = 0; w < windows; ++w) {
      if (!want(t, w, 0)) continue;
      Req reqs[3];
      int n = 0;
      if (t == Type::Tube) for (int r : tube_window(w)) reqs[n++] = Req{r, per_rank};
      else for (int r : plate_window(w)) reqs[n++] = Req{r, per_rank};
      Hand acc;
      expand(reqs, n, 0, v.wilds, 0, -1, acc, [&](const Hand& cards, int wl) {
        add(t, w, 0, -1, cards, wl);
      });
    }
  }

  void run() {
    singles();
    same_rank(Type::Pair, 2);
    same_rank(Type::Triple, 3);
    for (int n = 4; n <= 10; ++n) same_rank(Type::Bomb, n);
    joker_bomb();
    full_houses();
    straights();
    sequences(Type::Tube, 2);
    sequences(Type::Plate, 3);
  }
};

// Keeps only the highest key per (card multiset, type). Bomb-class types
// compare by strength, not by the raw key.
void prune_dominated(std::vector<Action>& out, size_t first) {
  if (out.size() - first < 2) return;
  std::vector<Action> kept;
  kept.reserve(out.size() - first);
  std::sort(out.begin() + first, out.end(), [](const Action& a, const Action& b) {
    if (a.cards.has1 != b.cards.has1) return a.cards.has1 < b.cards.has1;
    if (a.cards.has2 != b.cards.has2) return a.cards.has2 < b.cards.has2;
    if (a.type != b.type) return a.type < b.type;
    const Strength sa = strength(a), sb = strength(b);
    if (is_bomb_class(a.type)) return sb < sa;    // strongest first
    return a.key > b.key;
  });
  for (size_t i = first; i < out.size(); ++i) {
    if (i > first) {
      const Action& p = out[i - 1];
      const Action& c = out[i];
      if (p.cards == c.cards && p.type == c.type) continue;   // dominated
    }
    kept.push_back(out[i]);
  }
  out.resize(first);
  out.insert(out.end(), kept.begin(), kept.end());
}

// Drops exact duplicates, which the enumeration can produce when the same
// multiset is reachable by more than one path.
void drop_duplicates(std::vector<Action>& out, size_t first) {
  std::sort(out.begin() + first, out.end(), [](const Action& a, const Action& b) {
    if (a.cards.has1 != b.cards.has1) return a.cards.has1 < b.cards.has1;
    if (a.cards.has2 != b.cards.has2) return a.cards.has2 < b.cards.has2;
    if (a.type != b.type) return a.type < b.type;
    if (a.key != b.key) return a.key < b.key;
    return a.bomb_size < b.bomb_size;
  });
  out.erase(std::unique(out.begin() + first, out.end(),
                        [](const Action& a, const Action& b) {
                          return a.cards == b.cards && a.type == b.type &&
                                 a.key == b.key && a.bomb_size == b.bomb_size;
                        }),
            out.end());
}

}  // namespace

std::array<uint16_t, kNumSuits> sf_relevant_mask(const HandView& v) {
  // A card of rank r and suit s is relevant when some straight window through r
  // can still be completed in suit s from this hand plus its wild cards.
  std::array<uint16_t, kNumSuits> nat{};
  for (int c = 0; c < 52; ++c)
    if (v.card_count[c]) nat[suit_of(static_cast<CardId>(c))] |=
        static_cast<uint16_t>(1u << rank_of(static_cast<CardId>(c)));
  std::array<uint16_t, kNumSuits> out{};
  for (int s = 0; s < kNumSuits; ++s) {
    for (int w = 0; w < kNumStraightWindows; ++w) {
      int missing = 0;
      for (int r : straight_window(w)) missing += ((nat[s] >> r) & 1u) ? 0 : 1;
      if (missing > v.wilds) continue;
      for (int r : straight_window(w)) out[s] |= static_cast<uint16_t>(1u << r);
    }
    out[s] = static_cast<uint16_t>(out[s] & nat[s]);
  }
  return out;
}

void generate_moves(const Hand& hand, int level, const Action& top,
                    const ActionConfig& acfg, const RuleConfig& rules,
                    std::vector<Action>& out) {
  const size_t first = out.size();
  const bool leading = top.is_pass();
  if (!leading) out.push_back(Action{});        // pass

  const HandView v = make_view(hand, level);
  Gen g{hand, v, level, wild_id(level), acfg, rules, top, leading, out, out.size()};
  for (int c = 0; c < 52; ++c)
    if (v.card_count[c] && static_cast<CardId>(c) != g.wid)
      g.nat_mask[suit_of(static_cast<CardId>(c))] |=
          static_cast<uint16_t>(1u << rank_of(static_cast<CardId>(c)));
  g.sf_mask = sf_relevant_mask(v);
  g.run();

  const size_t body = first + (leading ? 0 : 1);
  drop_duplicates(out, body);
  if (acfg.prune_dominated_readings) prune_dominated(out, body);
}

void generate_tribute(const Hand& hand, int level, const RuleConfig&,
                      std::vector<Action>& out) {
  const HandView v = make_view(hand, level);
  const CardId wid = wild_id(level);
  int best = -1;
  for (int c = 0; c < kNumCardIds; ++c) {
    if (!v.card_count[c] || static_cast<CardId>(c) == wid) continue;
    best = std::max(best, power(rank_of(static_cast<CardId>(c)), level));
  }
  if (best < 0) return;
  for (int c = 0; c < kNumCardIds; ++c) {
    if (!v.card_count[c] || static_cast<CardId>(c) == wid) continue;
    if (power(rank_of(static_cast<CardId>(c)), level) != best) continue;
    Action a;
    a.type = Type::Tribute;
    a.key = static_cast<int8_t>(best);
    a.cards.add(static_cast<CardId>(c));
    out.push_back(a);
  }
}

void generate_back_tribute(const Hand& hand, int level, const RuleConfig& rules,
                           std::vector<Action>& out) {
  const HandView v = make_view(hand, level);
  const size_t first = out.size();
  for (int c = 0; c < 52; ++c) {
    if (!v.card_count[c]) continue;
    const int rank = rank_of(static_cast<CardId>(c));
    if (rank > 8) continue;                       // natural rank 2 to 10
    if (rank == level && !rules.back_tribute_level_cards) continue;
    Action a;
    a.type = Type::BackTribute;
    a.key = static_cast<int8_t>(power(rank, level));
    a.cards.add(static_cast<CardId>(c));
    out.push_back(a);
  }
  if (out.size() > first || !rules.back_tribute_fallback) return;
  // Nothing qualified: offer every card of the lowest power instead.
  int low = 99;
  for (int c = 0; c < kNumCardIds; ++c)
    if (v.card_count[c]) low = std::min(low, power(rank_of(static_cast<CardId>(c)), level));
  for (int c = 0; c < kNumCardIds; ++c) {
    if (!v.card_count[c]) continue;
    if (power(rank_of(static_cast<CardId>(c)), level) != low) continue;
    Action a;
    a.type = Type::BackTribute;
    a.key = static_cast<int8_t>(low);
    a.cards.add(static_cast<CardId>(c));
    out.push_back(a);
  }
}

bool is_legal(const Hand& hand, int level, const Action& top, const Action& a,
              const RuleConfig& rules) {
  if (a.is_pass()) return !top.is_pass();
  if (!hand.contains(a.cards)) return false;
  if (!beats(a, top)) return false;
  for (const auto& r : interpret(a.cards, level, rules))
    if (r.type == a.type && r.key == a.key && r.bomb_size == a.bomb_size) return true;
  return false;
}

}  // namespace gd
