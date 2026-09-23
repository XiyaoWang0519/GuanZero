// Rule profiles. Every item of docs/RULES.md section 13 sits behind a field
// here. `house` is the owner's ruleset and the default for training, internal
// evaluation and human play. `ogd` mirrors the OpenGuanDan simulator and is
// used only by the differential tests; see docs/ogd_profile_findings.md for
// the evidence behind each ogd value.
#pragma once

#include <cstdint>

namespace gd {

// Who pays whom in a double tribute (RULES.md 9.2, O1).
enum class TributePairing : uint8_t {
  Power = 0,        // house: the higher card goes to the Banker
  SeatGeometry = 1, // ogd: the Banker's downstream seat always pays the Banker
};
// Who pays the Banker and who leads when the two tribute cards have equal
// power (O1). Under SeatGeometry pairing only the leader is affected.
enum class TributeTie : uint8_t {
  Upstream = 0,     // pre-2026-09-22 house rule: (B + 3) % 4 pays B and leads
  LastFinisher = 1, // ogd: the seat recorded last in the finishing order leads
  Downstream = 2,   // house, official rule: clockwise tribute, (B + 1) % 4
                    // pays B and leads, (B + 3) % 4 pays the Follower
};
// How the last two seats of a double win are recorded (O5, logging only).
enum class DoubleWinTail : uint8_t {
  NextFollower = 0,  // house: by seat from next(Follower)
  AscendingSeat = 1, // ogd: by ascending seat index
};
enum class ActionMode : uint8_t { Full = 0, Canonical = 1 };
enum class WildUsage : uint8_t { Minimal = 0, All = 1 };
enum class FirstLeader : uint8_t { Random = 0, Fixed = 1 };

struct RuleConfig {
  // O1
  TributePairing tribute_pairing = TributePairing::Power;
  TributeTie tribute_tie = TributeTie::Downstream;
  // O2: level cards may not be returned as back-tribute, in any suit.
  bool back_tribute_level_cards = false;
  // house falls back to any lowest-power card when nothing qualifies; the
  // simulator has no fallback path at all.
  bool back_tribute_fallback = true;
  // O3
  bool full_house_joker_pair = true;
  // O4
  bool pass_a_requires_owner = true;
  // A failed attempt is counted when the owner wins the round with Banker and
  // Dweller. house also counts a plain loss of an owned A round; ogd does not.
  bool a_fail_on_loss = true;
  // house follows the official rules: no reset, a team at A plays A until it
  // passes. 3 is the Nanjing / 翻山 variant (reset to the deuce).
  int a_fail_limit = 0;          // 0 disables the reset
  int a_fail_reset_level = 0;    // rank index, 0 == the deuce
  // ogd ends the match by victory count after this many resets; 0 disables.
  int shuffle_limit = 0;
  // O5
  DoubleWinTail double_win_tail = DoubleWinTail::NextFollower;
  // O10
  bool tribute_between_partners = true;
  // O6
  FirstLeader first_leader = FirstLeader::Random;
  int fixed_first_leader = 0;

  static RuleConfig house() { return RuleConfig{}; }

  // Evidence: docs/ogd_profile_findings.md. Items still open at M0 are marked
  // there; they are settled by the replay diff, not guessed at here.
  static RuleConfig ogd() {
    RuleConfig c;
    c.tribute_pairing = TributePairing::SeatGeometry;
    c.tribute_tie = TributeTie::LastFinisher;
    c.back_tribute_fallback = false;
    c.a_fail_on_loss = false;
    c.a_fail_limit = 4;
    c.shuffle_limit = 50;
    c.double_win_tail = DoubleWinTail::AscendingSeat;
    return c;
  }
};

// The three canonical-mode reductions of docs/RULES.md 11.2, each behind its
// own flag.
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
