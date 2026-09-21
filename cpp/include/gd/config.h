// Rule profiles. Every item of docs/RULES.md section 13 sits behind a field here.
#pragma once

#include <cstdint>

namespace gd {

enum class TributeTie : uint8_t { Upstream = 0, Downstream = 1 };
enum class ActionMode : uint8_t { Full = 0, Canonical = 1 };
enum class WildUsage : uint8_t { Minimal = 0, All = 1 };
enum class FirstLeader : uint8_t { Random = 0, Fixed = 1 };

struct RuleConfig {
  // O1
  TributeTie tribute_tie = TributeTie::Upstream;
  // O2: level cards may not be returned as back-tribute
  bool back_tribute_level_cards = false;
  // O3
  bool full_house_joker_pair = true;
  // O4
  bool pass_a_requires_owner = true;
  int a_fail_limit = 3;          // 0 disables the reset
  int a_fail_reset_level = 0;    // rank index, 0 == the deuce
  // O10
  bool tribute_between_partners = true;
  // O6
  FirstLeader first_leader = FirstLeader::Random;
  int fixed_first_leader = 0;

  static RuleConfig house() { return RuleConfig{}; }
  static RuleConfig ogd();   // filled in by M0 trace replay, see docs/RULES.md 13
};

// The three canonical-mode reductions of docs/RULES.md 11.2, each behind its own flag.
struct ActionConfig {
  ActionMode mode = ActionMode::Canonical;
  bool prune_dominated_readings = true;
  bool suit_dedup = true;
  WildUsage wild_usage = WildUsage::Minimal;

  static ActionConfig full() {
    ActionConfig c;
    c.mode = ActionMode::Full;
    c.prune_dominated_readings = false;
    c.suit_dedup = false;
    c.wild_usage = WildUsage::All;
    return c;
  }
};

}  // namespace gd
