// In-engine heuristic players. See docs/DESIGN.md 8.2 item 4 and 8.3 item 1.
#include "gd/bots.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gd/movegen.h"

namespace gd {
namespace {

uint64_t next_random(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// How reluctant we are to make this play. Lower is played sooner.
// Bombs and straight flushes are hoarded, long plays are preferred over short
// ones at equal key, and a high key is spent late.
int64_t play_cost(const Action& a) {
  int64_t cost = 0;
  if (is_bomb_class(a.type)) cost += 1'000'000;
  if (a.type == Type::JokerBomb) cost += 1'000'000;
  cost += int64_t(a.wilds) * 20'000;          // wild cards are scarce
  cost += int64_t(a.key) * 100;               // spend low cards first
  cost -= int64_t(a.cards.size()) * 10;       // shed more cards when equal
  return cost;
}

bool empties_hand(const MatchState& m, const Action& a) {
  return a.cards.size() == m.round.hands[m.round.to_move].size();
}

// The smallest number of cards any opponent still holds.
int opponent_pressure(const MatchState& m) {
  const int self = m.round.to_move;
  int least = std::numeric_limits<int>::max();
  for (int s = 0; s < 4; ++s) {
    if (s == self || s == (self + 2) % 4) continue;
    if (!m.round.active(s)) continue;
    least = std::min<int>(least, m.round.hands[s].size());
  }
  return least == std::numeric_limits<int>::max() ? 99 : least;
}

// How high this play sits inside its own ordering, normalised to [0, 1].
// Single, Pair, Triple, FullHouse and Bomb are keyed by power; Straight, Tube,
// Plate and StraightFlush are keyed by the sequence window (RULES.md 4). The
// two orderings are never mixed: each type is normalised by its own span.
double play_rank(const Action& a) {
  if (is_bomb_class(a.type)) return 1.0;   // a bomb is overwhelming by class
  switch (a.type) {
    case Type::Single:
    case Type::Pair:
    case Type::Triple:
    case Type::FullHouse:
      return double(a.key) / double(kNumPowers - 1);
    case Type::Straight:
      return double(a.key) / double(kNumStraightWindows - 1);
    case Type::Tube:
      return double(a.key) / double(kNumTubeWindows - 1);
    case Type::Plate:
      return double(a.key) / double(kNumPlateWindows - 1);
    default:
      return 0.0;
  }
}

// Weights of the style terms, in internal score units. StyleParams::kScoreUnit
// is one logit, so a type preference of 1 is one logit by construction; the
// lead and follow weights are deliberately smaller and larger than the key term
// of play_cost so that the biases can reorder plays without ever outranking
// emptying the hand.
constexpr double kLeadWeight = 5'000.0;
constexpr double kFollowWeight = 60'000.0;
constexpr double kOutBonus = 10'000'000.0;

}  // namespace

int random_bot(const MatchState&, const std::vector<Action>& cands, uint64_t& rng) {
  if (cands.empty()) return 0;
  return static_cast<int>(next_random(rng) % cands.size());
}

int greedy_bot(const MatchState& m, const std::vector<Action>& cands, uint64_t& rng) {
  if (cands.empty()) return 0;
  if (m.round.phase != Phase::Play) return tribute_bot(m, cands, rng);

  const int self = m.round.to_move;
  const int partner = (self + 2) % 4;
  const bool leading = m.round.top.is_pass();
  const bool partner_holds_top = !leading && m.round.holder == partner;
  const int pressure = opponent_pressure(m);
  // An opponent on the brink is worth a bomb; so is going out ourselves.
  const bool urgent = pressure <= 2;

  int pass_index = -1;
  int best = -1;
  int64_t best_cost = std::numeric_limits<int64_t>::max();

  for (size_t i = 0; i < cands.size(); ++i) {
    const Action& a = cands[i];
    if (a.is_pass()) { pass_index = static_cast<int>(i); continue; }

    const bool out = empties_hand(m, a);
    // Let the partner's play stand unless we can finish on it.
    if (partner_holds_top && !out) continue;
    // Hoard the bomb class: spend it only to go out or to stop a threat.
    if (is_bomb_class(a.type) && !out && !urgent) continue;

    int64_t cost = play_cost(a);
    if (out) cost -= 10'000'000;              // going out beats everything
    if (cost < best_cost) { best_cost = cost; best = static_cast<int>(i); }
  }

  if (best >= 0) return best;
  if (pass_index >= 0) return pass_index;
  // Following with nothing cheap left: take the least costly legal play.
  best = 0;
  best_cost = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < cands.size(); ++i) {
    const int64_t cost = play_cost(cands[i]);
    if (cost < best_cost) { best_cost = cost; best = static_cast<int>(i); }
  }
  return best;
}

int styled_bot(const MatchState& m, const std::vector<Action>& cands,
               const StyleParams& style, uint64_t& rng) {
  if (cands.empty()) return 0;
  if (m.round.phase != Phase::Play) return tribute_bot(m, cands, rng);

  const int self = m.round.to_move;
  const int partner = (self + 2) % 4;
  const bool leading = m.round.top.is_pass();
  const bool partner_holds_top = !leading && m.round.holder == partner;
  // The sentinel of a round where no opponent is still active is clamped to a
  // full hand, so that the bomb gate spans the whole reachable range.
  const int pressure = std::min(opponent_pressure(m), kHandSize);
  const int partner_cards = m.round.hands[partner].size();

  // bomb_threshold 1 unlocks the bomb class only at pressure <= 2, which is the
  // greedy rule; 0 unlocks it at every pressure.
  const double bomb_limit =
      double(1.0f - style.bomb_threshold()) * double(kHandSize) + 2.0;
  // partner_weight 1 demands more cards than a hand can hold, so the partner is
  // never overtaken; 0 demands none.
  const double partner_gate =
      double(style.partner_weight()) * double(kHandSize + 1);

  const auto allowed = [&](const Action& a, bool out) {
    if (out) return true;                     // going out is always allowed
    if (partner_holds_top && !(double(partner_cards) >= partner_gate)) return false;
    if (is_bomb_class(a.type) && !(double(pressure) <= bomb_limit)) return false;
    return true;
  };

  const auto score_of = [&](const Action& a, bool out) {
    double s = -double(play_cost(a));
    if (out) s += kOutBonus;
    if (leading) {
      s += double(style.type_pref(a.type)) * StyleParams::kScoreUnit;
      s += double(style.lead_high_bias()) * kLeadWeight * play_rank(a);
    } else {
      s += double(style.follow_aggression()) * kFollowWeight * play_rank(a);
    }
    return s;
  };

  // Pass 1: the admissible set and its best score. A pass is never scored; like
  // greedy_bot, the styled bot plays whenever anything is admissible.
  int pass_index = -1;
  int best = -1;
  double best_score = -std::numeric_limits<double>::infinity();
  int considered = 0;
  for (size_t i = 0; i < cands.size(); ++i) {
    const Action& a = cands[i];
    if (a.is_pass()) { pass_index = static_cast<int>(i); continue; }
    const bool out = empties_hand(m, a);
    if (!allowed(a, out)) continue;
    ++considered;
    const double s = score_of(a, out);
    if (s > best_score) { best_score = s; best = static_cast<int>(i); }
  }

  // Nothing admissible: pass if we may, otherwise take the best of everything.
  bool fallback = false;
  if (best < 0) {
    if (pass_index >= 0) return pass_index;
    fallback = true;
    considered = 0;
    for (size_t i = 0; i < cands.size(); ++i) {
      const double s = score_of(cands[i], empties_hand(m, cands[i]));
      ++considered;
      if (s > best_score) { best_score = s; best = static_cast<int>(i); }
    }
  }
  if (best < 0) return 0;

  const double temp = double(style.temperature());
  if (!(temp > 0.0) || considered < 2) return best;

  // Softmax over the same scores, in logit units. Two more passes keep the hot
  // loop free of allocation: one for the partition, one for the draw.
  const double scale = StyleParams::kScoreUnit * temp;
  const auto weight = [&](size_t i) {
    const Action& a = cands[i];
    const bool out = empties_hand(m, a);
    const double e = (score_of(a, out) - best_score) / scale;
    return e < -80.0 ? 0.0 : std::exp(e);
  };
  const auto usable = [&](size_t i) {
    if (fallback) return true;
    const Action& a = cands[i];
    if (a.is_pass()) return false;
    return allowed(a, empties_hand(m, a));
  };

  double total = 0.0;
  for (size_t i = 0; i < cands.size(); ++i)
    if (usable(i)) total += weight(i);
  if (!(total > 0.0)) return best;

  const double draw =
      double(next_random(rng) >> 11) * (1.0 / 9007199254740992.0) * total;
  double acc = 0.0;
  for (size_t i = 0; i < cands.size(); ++i) {
    if (!usable(i)) continue;
    acc += weight(i);
    if (draw < acc) return static_cast<int>(i);
  }
  return best;
}

int tribute_bot(const MatchState& m, const std::vector<Action>& cands, uint64_t& rng) {
  if (cands.empty()) return 0;
  if (cands.size() == 1) return 0;

  const int self = m.round.to_move;
  const Hand& hand = m.round.hands[self];
  const HandView view = make_view(hand, m.round.level);
  const auto sf_mask = sf_relevant_mask(view);

  if (cands[0].type == Type::Tribute) {
    // Among tied top cards, give one whose suit is not straight-flush relevant.
    int best = 0;
    bool best_sf = true;
    for (size_t i = 0; i < cands.size(); ++i) {
      const CardId c = static_cast<CardId>(std::countr_zero(cands[i].cards.has1));
      const int suit = suit_of(c);
      const bool sf = suit >= 0 && (sf_mask[suit] >> rank_of(c)) & 1u;
      if (!sf && best_sf) { best = static_cast<int>(i); best_sf = false; }
    }
    return best;
  }

  // Back-tribute. Giving to the partner after a double win means giving a good
  // card; giving to an opponent means giving the weakest card we can spare.
  bool to_partner = false;
  for (int i = 0; i < m.round.num_tribute_moves; ++i) {
    const TributeMove& t = m.round.tribute_moves[i];
    if (!t.back && t.receiver == self)
      to_partner = (t.payer == (self + 2) % 4);
  }

  int best = 0;
  int64_t best_score = std::numeric_limits<int64_t>::min();
  for (size_t i = 0; i < cands.size(); ++i) {
    const CardId c = static_cast<CardId>(std::countr_zero(cands[i].cards.has1));
    const int rank = rank_of(c);
    const int suit = suit_of(c);
    const int copies = view.rank_count[rank];
    int64_t penalty = 0;
    if (copies == 2) penalty += 300;                 // breaks a pair
    if (copies == 3) penalty += 600;                 // breaks a triple
    if (copies >= 4) penalty += 5000;                // breaks a bomb
    if (suit >= 0 && ((sf_mask[suit] >> rank) & 1u)) penalty += 400;
    if (rank == 3 || rank == 8) penalty += 200;      // every straight holds a 5 or a 10
    const int64_t value = power(rank, m.round.level);
    // To the partner: the highest card we can spare. To an opponent: the lowest.
    const int64_t score = (to_partner ? value * 10 : -value * 10) - penalty;
    if (score > best_score) { best_score = score; best = static_cast<int>(i); }
  }
  (void)rng;
  return best;
}

}  // namespace gd
