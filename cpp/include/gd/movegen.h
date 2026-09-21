// Legal move generation, both enumeration modes. See docs/RULES.md 11.2.
#pragma once

#include <vector>

#include "gd/action.h"
#include "gd/cards.h"
#include "gd/config.h"

namespace gd {

// Appends every legal action for `hand` at `level` over `top` into `out`.
// A `top` of type Pass means the seat is leading, in which case no pass is
// emitted; otherwise pass is emitted first.
// Canonical mode must always produce a subset of full mode that preserves the
// best reading of every playable multiset (RULES.md 11.2 and 14.5).
void generate_moves(const Hand& hand, int level, const Action& top,
                    const ActionConfig& acfg, const RuleConfig& rules,
                    std::vector<Action>& out);

// Tribute and back-tribute candidates (RULES.md 9.3). Emitted as single-card
// actions of type Tribute / BackTribute so that they share the batch path.
void generate_tribute(const Hand& hand, int level, const RuleConfig& rules,
                      std::vector<Action>& out);
void generate_back_tribute(const Hand& hand, int level, const RuleConfig& rules,
                           std::vector<Action>& out);

// True if `a` is a legal play of `hand` over `top` at `level`.
bool is_legal(const Hand& hand, int level, const Action& top, const Action& a,
              const RuleConfig& rules);

// A card of rank r and suit s is SF-relevant if some straight window containing
// r could still become a straight flush in suit s from this hand plus its wilds.
// Used by the suit_dedup reduction and by the tribute features.
std::array<uint16_t, kNumSuits> sf_relevant_mask(const HandView& v);

}  // namespace gd
