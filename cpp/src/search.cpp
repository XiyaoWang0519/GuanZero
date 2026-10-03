#include "gd/search.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <random>
#include <stdexcept>
#include <vector>

namespace gd {

namespace {

struct Residual {
  std::array<Hand, 4> sampled;         // observer's hand and publicly known holdings
  std::vector<CardId> pool;            // unseen copies still to deal
};

// Public constraints shared by every sampler: the observer's own hand, played
// cards, public tribute transfers and anti-tribute red-joker holders. Reads no
// other seat's current cards.
Residual residual_cards(const MatchState& source, int observer) {
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
  return Residual{sampled, std::move(pool)};
}

}  // namespace

MatchState determinize_uniform(const MatchState& source, int observer, uint64_t seed) {
  Residual residual = residual_cards(source, observer);
  std::array<Hand, 4>& sampled = residual.sampled;
  std::vector<CardId>& pool = residual.pool;
  const RoundState& round = source.round;
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

MatchState determinize_weighted(const MatchState& source, int observer, uint64_t seed,
                                const SeatWeights& weights) {
  for (const auto& row : weights)
    for (const float w : row)
      if (!(w >= 0.0f) || !std::isfinite(w))
        throw std::invalid_argument("seat weights must be finite and nonnegative");
  Residual residual = residual_cards(source, observer);
  std::array<Hand, 4>& sampled = residual.sampled;
  std::vector<CardId>& pool = residual.pool;
  const RoundState& round = source.round;
  std::array<int, 3> room{};
  for (int r = 1; r <= 3; ++r) {
    const int seat = (observer + r) % 4;
    room[r - 1] = round.hands[seat].size() - sampled[seat].size();
  }
  std::mt19937_64 rng(seed);
  std::shuffle(pool.begin(), pool.end(), rng);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (const CardId card : pool) {
    double total = 0.0;
    std::array<double, 3> w{};
    for (int r = 0; r < 3; ++r) {
      w[r] = room[r] > 0 ? static_cast<double>(weights[r][card]) : 0.0;
      total += w[r];
    }
    if (total <= 0.0) {   // no weight on a seat with room: uniform over the seats with room
      for (int r = 0; r < 3; ++r) w[r] = room[r] > 0 ? 1.0 : 0.0, total += w[r];
    }
    double draw = unit(rng) * total;
    int chosen = -1;
    for (int r = 0; r < 3; ++r) {
      if (w[r] <= 0.0) continue;
      chosen = r;
      draw -= w[r];
      if (draw < 0.0) break;
    }
    sampled[(observer + chosen + 1) % 4].add(card);
    --room[chosen];
  }

  MatchState result = source;
  result.round.hands = sampled;
  return result;
}

}  // namespace gd
