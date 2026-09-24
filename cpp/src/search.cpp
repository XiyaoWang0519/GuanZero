#include "gd/search.h"

#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>
#include <vector>

namespace gd {

MatchState determinize_uniform(const MatchState& source, int observer, uint64_t seed) {
  if (source.round.phase != Phase::Play)
    throw std::invalid_argument("determinization requires the play phase");
  if (observer < 0 || observer >= 4)
    throw std::out_of_range("observer outside seats");

  const RoundState& round = source.round;
  std::array<int, kNumCardIds> remaining{};
  for (int c = 0; c < kNumCardIds; ++c) {
    remaining[c] = 2 - round.hands[observer].count(static_cast<CardId>(c));
    for (const Hand& played : round.played)
      remaining[c] -= played.count(static_cast<CardId>(c));
    if (remaining[c] < 0) throw std::invalid_argument("invalid public card multiplicity");
  }

  std::array<Hand, 4> sampled{};
  sampled[observer] = round.hands[observer];
  const int red_joker_holders = std::count(round.held_red_joker.begin(),
                                           round.held_red_joker.end(), true);
  // Public transfers give each receiver a lower bound until those copies are
  // publicly played. Match the observation encoder's ordered transfer rule.
  for (int seat = 0; seat < 4; ++seat) {
    if (seat == observer) continue;
    std::array<int, kNumCardIds> known{};
    for (int i = 0; i < round.num_tribute_moves; ++i) {
      const TributeMove& move = round.tribute_moves[i];
      if (move.payer == seat) known[move.card] = std::max(0, known[move.card] - 1);
      if (move.receiver == seat) ++known[move.card];
    }
    if (round.anti_tribute && round.held_red_joker[seat])
      known[kRJ] = std::max(known[kRJ], red_joker_holders == 1 ? 2 : 1);
    for (int c = 0; c < kNumCardIds; ++c) {
      const int count = std::max(0, known[c] - round.played[seat].count(static_cast<CardId>(c)));
      if (count > remaining[c])
        throw std::invalid_argument("public known holdings exceed unseen copies");
      sampled[seat].add_n(static_cast<CardId>(c), count);
      remaining[c] -= count;
    }
    if (sampled[seat].size() > round.hands[seat].size())
      throw std::invalid_argument("public known holdings exceed seat size");
  }
  // Anti-tribute announces exactly which seats held the red jokers. No cards
  // are exchanged, so none may be assigned to an unannounced seat.
  if (round.anti_tribute && remaining[kRJ] != 0)
    throw std::invalid_argument("anti-tribute red jokers conflict with public holders");

  std::vector<CardId> pool;
  for (int c = 0; c < kNumCardIds; ++c)
    for (int i = 0; i < remaining[c]; ++i) pool.push_back(static_cast<CardId>(c));
  int needed = 0;
  for (int seat = 0; seat < 4; ++seat)
    if (seat != observer) needed += round.hands[seat].size() - sampled[seat].size();
  if (needed != static_cast<int>(pool.size()))
    throw std::invalid_argument("public hand sizes do not match unseen cards");

  std::mt19937_64 rng(seed);
  std::shuffle(pool.begin(), pool.end(), rng);
  size_t at = 0;
  for (int seat = 0; seat < 4; ++seat) {
    if (seat == observer) continue;
    const int count = round.hands[seat].size() - sampled[seat].size();
    for (int i = 0; i < count; ++i) sampled[seat].add(pool[at++]);
  }

  MatchState result = source;
  result.round.hands = sampled;
  return result;
}

}  // namespace gd
