// In-engine heuristic players. Used to bootstrap self-play out of random play
// (DESIGN.md 8.2 item 4), as league opponents and as the fuzzer's driver.
#pragma once

#include <cstdint>
#include <vector>

#include "gd/state.h"

namespace gd {

// Uniform over the candidate list.
int random_bot(const MatchState& m, const std::vector<Action>& cands, uint64_t& rng);

// A greedy player. Leads its cheapest play, follows with the cheapest play that
// beats the top, never spends a bomb unless it is the only option or an
// opponent is about to finish, and never overtakes its own partner.
int greedy_bot(const MatchState& m, const std::vector<Action>& cands, uint64_t& rng);

// Tribute and back-tribute heuristics of DESIGN.md 8.3 item 1.
int tribute_bot(const MatchState& m, const std::vector<Action>& cands, uint64_t& rng);

}  // namespace gd
