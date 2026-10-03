// Information-set sampling for bounded endgame search.
#pragma once

#include <array>
#include <cstdint>

#include "gd/state.h"

namespace gd {

// Clone a play state and shuffle residual unseen cards into other seats.
// The sampler reads only the observer's hand and public state from `source`:
// played cards, hand sizes, and public tribute/anti-tribute records. It never
// reads the identities of another seat's current cards. This is a uniform
// shuffle of residual copies, not a calibrated posterior over legal hands.
MatchState determinize_uniform(const MatchState& source, int observer, uint64_t seed);

// Per-card seat weights for the weighted sampler: weights[r - 1][card] is the
// (unnormalised, nonnegative) weight that an unseen copy of `card` sits with
// the seat r places after the observer (r = 1 next to play, 2 partner, 3
// previous). The layout matches the actor's hidden-hand belief head.
using SeatWeights = std::array<std::array<float, kNumCardIds>, 3>;

// Like determinize_uniform (same public constraints, same information
// boundary), but the residual copies are dealt one by one in a seeded random
// order, each to a seat drawn in proportion to its weight for that card among
// the seats that still have room. Seats with no weight left fall back to a
// uniform draw over the seats with room, so every sample is a legal deal.
// Weights must be finite and nonnegative. With equal weights this is a
// uniform sample (though not the same one as determinize_uniform for a seed).
MatchState determinize_weighted(const MatchState& source, int observer, uint64_t seed,
                                const SeatWeights& weights);

}  // namespace gd
