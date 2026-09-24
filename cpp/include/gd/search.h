// Information-set sampling for bounded endgame search.
#pragma once

#include <cstdint>

#include "gd/state.h"

namespace gd {

// Clone a play state and shuffle residual unseen cards into other seats.
// The sampler reads only the observer's hand and public state from `source`:
// played cards, hand sizes, and public tribute/anti-tribute records. It never
// reads the identities of another seat's current cards. This is a uniform
// shuffle of residual copies, not a calibrated posterior over legal hands.
MatchState determinize_uniform(const MatchState& source, int observer, uint64_t seed);

}  // namespace gd
