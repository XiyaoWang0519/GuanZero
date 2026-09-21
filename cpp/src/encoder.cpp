// Observation and action encoding, v1. The layout contract is gd/encoder.h.
#include "gd/encoder.h"

#include <algorithm>
#include <array>

#include "gd/movegen.h"
#include "gd/rules.h"

namespace gd {
namespace {

constexpr int partner_of(int s) { return (s + 2) % 4; }
// rel 1 is the next seat to play, 2 the partner, 3 the previous seat.
constexpr int rel_of(int seat, int other) { return (other - seat + 4) % 4; }
constexpr int seat_at(int seat, int rel) { return (seat + rel) % 4; }

void put_cards(const Hand& h, std::span<float> dst, int off) {
  uint64_t b = h.has1;
  while (b) {
    const int c = std::countr_zero(b);
    b &= b - 1;
    dst[off + c] = 1.0f;
  }
  b = h.has2;
  while (b) {
    const int c = std::countr_zero(b);
    b &= b - 1;
    dst[off + kNumCardIds + c] = 1.0f;
  }
}

void one_hot(std::span<float> dst, int off, int index, int size) {
  if (index < 0 || index >= size) return;
  dst[off + index] = 1.0f;
}

// The twelve wild-card flags documented in gd/encoder.h.
void wild_flags(const HandView& v, int level, std::span<float> dst, int off) {
  const int w = v.wilds;
  const auto& rc = v.rank_count;
  auto set = [&](int i) { dst[off + i] = 1.0f; };

  if (w > 0) {
    for (int r = 0; r < 13; ++r) {
      if (rc[r] >= 1 && rc[r] < 2 && rc[r] + w >= 2) set(0);
      if (rc[r] >= 1 && rc[r] < 3 && rc[r] + w >= 3) set(1);
      if (rc[r] >= 1 && rc[r] < 4 && rc[r] + w >= 4) set(6);
      for (int n = 5; n <= 10; ++n)
        if (rc[r] >= 1 && rc[r] < n && rc[r] + w >= n) { set(7); break; }
    }
    // Full house: a triple and a pair of different ranks, short by at most w.
    for (int a = 0; a < 13 && !dst[off + 2]; ++a) {
      for (int b = 0; b < 13; ++b) {
        if (a == b) continue;
        const int need = std::max(0, 3 - rc[a]) + std::max(0, 2 - rc[b]);
        if (need >= 1 && need <= w && rc[a] + rc[b] > 0) { set(2); break; }
      }
    }
    for (int i = 0; i < kNumStraightWindows; ++i) {
      int missing = 0;
      for (int r : straight_window(i)) missing += rc[r] ? 0 : 1;
      if (missing >= 1 && missing <= w) set(3);
    }
    for (int i = 0; i < kNumTubeWindows; ++i) {
      int need = 0;
      for (int r : tube_window(i)) need += std::max(0, 2 - rc[r]);
      if (need >= 1 && need <= w) set(4);
    }
    for (int i = 0; i < kNumPlateWindows; ++i) {
      int need = 0;
      for (int r : plate_window(i)) need += std::max(0, 3 - rc[r]);
      if (need >= 1 && need <= w) set(5);
    }
    std::array<uint16_t, kNumSuits> nat{};
    for (int c = 0; c < 52; ++c)
      if (v.card_count[c] && c != wild_id(level))
        nat[suit_of(static_cast<CardId>(c))] |=
            static_cast<uint16_t>(1u << rank_of(static_cast<CardId>(c)));
    for (int s = 0; s < kNumSuits; ++s)
      for (int i = 0; i < kNumStraightWindows; ++i) {
        int missing = 0;
        for (int r : straight_window(i)) missing += ((nat[s] >> r) & 1u) ? 0 : 1;
        if (missing >= 1 && missing <= w) { set(8); break; }
      }
  }
  if (rc[level] > 0) set(9);          // a natural level card, the wild excluded
  if (w >= 2) set(10);
  if (dst[off + 6] > 0.0f || dst[off + 7] > 0.0f) set(11);
}

}  // namespace

void encode_action(const Action& a, const RoundState& r, int seat,
                   std::span<float> dst) {
  std::fill(dst.begin(), dst.end(), 0.0f);
  put_cards(a.cards, dst, kActCards1);
  one_hot(dst, kActType, static_cast<int>(a.type), static_cast<int>(Type::kNumTypes));
  one_hot(dst, kActKey, a.key, kNumPowers);
  if (a.type == Type::Bomb) one_hot(dst, kActBombSize, a.bomb_size - 4, 7);
  one_hot(dst, kActWilds, std::clamp<int>(a.wilds, 0, 2), 3);

  if (a.type != Type::Tribute && a.type != Type::BackTribute) return;
  if (a.cards.empty()) return;
  const CardId c = a.cards.to_vector().front();
  const Hand& hand = r.hands[seat];
  const HandView v = make_view(hand, r.level);
  const auto sf = sf_relevant_mask(v);
  const int rank = rank_of(c);
  const int suit = suit_of(c);
  const int copies = hand.count(c);
  const int rc = rank >= kRankBJ ? v.card_count[c] : v.rank_count[rank];
  float* f = dst.data() + kActTributeFlags;
  f[0] = copies == 1 ? 1.0f : 0.0f;
  f[1] = copies == 2 ? 1.0f : 0.0f;
  f[2] = rc == 2 ? 1.0f : 0.0f;                     // giving it breaks a pair
  f[3] = rc == 3 ? 1.0f : 0.0f;                     // breaks a triple
  f[4] = rc >= 4 ? 1.0f : 0.0f;                     // breaks a bomb
  f[5] = (suit >= 0 && ((sf[suit] >> rank) & 1u)) ? 1.0f : 0.0f;
  f[6] = rank == 3 ? 1.0f : 0.0f;                   // a natural five
  f[7] = rank == 8 ? 1.0f : 0.0f;                   // a natural ten
}

void encode_observation(const MatchState& m, int seat, std::span<float> dst) {
  std::fill(dst.begin(), dst.end(), 0.0f);
  const RoundState& r = m.round;
  const Hand& own = r.hands[seat];
  const int level = r.level;

  put_cards(own, dst, kObsOwnHand);

  // Unseen: two copies of every card minus our hand minus everything played.
  Hand unseen;
  for (int c = 0; c < kNumCardIds; ++c) {
    int n = 2 - own.count(static_cast<CardId>(c));
    for (int s = 0; s < 4; ++s) n -= r.played[s].count(static_cast<CardId>(c));
    for (int k = 0; k < n; ++k) unseen.add(static_cast<CardId>(c));
  }
  put_cards(unseen, dst, kObsUnseen);

  for (int rel = 1; rel <= 3; ++rel)
    put_cards(r.played[seat_at(seat, rel)], dst, kObsPlayed + (rel - 1) * 108);
  put_cards(r.played[seat], dst, kObsPlayed + 3 * 108);

  for (int rel = 1; rel <= 3; ++rel)
    one_hot(dst, kObsCardsLeft + (rel - 1) * 28,
            r.hands[seat_at(seat, rel)].size(), 28);

  for (int k = 0; k < 4; ++k) {
    const int s = seat_at(seat, k);
    if (r.finish_pos[s] >= 0) one_hot(dst, kObsFinishStatus + k * 4, r.finish_pos[s], 4);
  }

  one_hot(dst, kObsLevels, level, 13);
  one_hot(dst, kObsLevels + 13, m.levels[seat % 2], 13);
  one_hot(dst, kObsLevels + 26, m.levels[1 - (seat % 2)], 13);

  const HandView v = make_view(own, level);
  one_hot(dst, kObsWildHeld, std::clamp(v.wilds, 0, 2), 3);
  int wild_unseen = 2 - own.count(wild_id(level));
  for (int s = 0; s < 4; ++s) wild_unseen -= r.played[s].count(wild_id(level));
  one_hot(dst, kObsWildUnseen, std::clamp(wild_unseen, 0, 2), 3);
  wild_flags(v, level, dst, kObsWildFlags);

  if (!r.top.is_pass()) {
    encode_action(r.top, r, seat,
                  dst.subspan(kObsTrickTop, kActDim));
    const int rel = r.holder >= 0 ? rel_of(seat, r.holder) : 0;
    one_hot(dst, kObsTrickHolder, rel, 4);
  } else {
    dst[kObsTrickLeading] = 1.0f;
  }
  one_hot(dst, kObsTrickPasses, std::clamp<int>(r.passes, 0, 3), 4);

  for (int rel = 1; rel <= 3; ++rel) {
    const int s = seat_at(seat, rel);
    if (!r.has_acted[s]) continue;
    encode_action(r.last_action[s], r, s,
                  dst.subspan(kObsLastAction + (rel - 1) * kActDim, kActDim));
  }

  int phase_index = -1;
  if (r.phase == Phase::Play) phase_index = 0;
  else if (r.phase == Phase::Tribute) phase_index = 1;
  else if (r.phase == Phase::BackTribute) phase_index = 2;
  one_hot(dst, kObsPhase, phase_index, 3);

  int role = 4;                                  // none, for example round one
  if (m.has_prev)
    for (int i = 0; i < 4; ++i) if (m.prev_order[i] == seat) role = i;
  one_hot(dst, kObsRoles, role, 5);
  for (int i = 0; i < r.num_tribute_moves; ++i) {
    const TributeMove& t = r.tribute_moves[i];
    if (t.payer == seat && t.receiver == partner_of(seat)) dst[kObsRoles + 5] = 1.0f;
    if (t.receiver == seat && t.payer == partner_of(seat)) dst[kObsRoles + 5] = 1.0f;
  }

  for (int i = 0; i < r.num_tribute_moves && i < 4; ++i) {
    const TributeMove& t = r.tribute_moves[i];
    const int off = kObsTribute + i * 62;
    one_hot(dst, off, t.card, 54);
    one_hot(dst, off + 54, rel_of(seat, t.payer), 4);
    one_hot(dst, off + 58, rel_of(seat, t.receiver), 4);
  }

  // Cards publicly known to sit in another seat's hand: what tribute gave them
  // and they have not played since.
  for (int rel = 1; rel <= 3; ++rel) {
    const int s = seat_at(seat, rel);
    for (int i = 0; i < r.num_tribute_moves; ++i) {
      const TributeMove& t = r.tribute_moves[i];
      if (t.receiver != s) continue;
      if (r.played[s].count(t.card)) continue;
      dst[kObsKnownHoldings + (rel - 1) * 54 + t.card] = 1.0f;
    }
  }
}

}  // namespace gd
