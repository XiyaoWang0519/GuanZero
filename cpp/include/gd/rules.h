// Reading a card multiset: the C++ counterpart of the oracle's interpret().
#pragma once

#include <vector>

#include "gd/action.h"
#include "gd/config.h"

namespace gd {

// A reading without cards: what the oracle returns as (type, key).
struct Reading {
  Type type = Type::Pass;
  int8_t key = 0;
  int8_t bomb_size = 0;
  int8_t fh_pair_rank = -1;
  constexpr bool operator==(const Reading& o) const {
    return type == o.type && key == o.key && bomb_size == o.bomb_size;
  }
};

// Every (type, key) reading of a card multiset at the given round level.
// Mirrors gd_reference.interpret. An empty multiset reads as Pass.
// Readings that differ only in the full house pair rank collapse to one entry
// per (type, key) here; movegen keeps the pair rank for the abstract id.
std::vector<Reading> interpret(const Hand& cards, int level,
                               const RuleConfig& rules = RuleConfig::house());

// Canonical mode's per-type reduction: the highest key per type.
std::vector<Reading> best_readings(const Hand& cards, int level,
                                   const RuleConfig& rules = RuleConfig::house());

// Round bookkeeping, mirroring the oracle.
// order lists seats by finishing position. Returns the winning team and its gain.
struct LevelGain { int team; int gain; };
LevelGain level_gain(const std::array<int8_t, 4>& order);
int promote(int level, int gain);

}  // namespace gd
