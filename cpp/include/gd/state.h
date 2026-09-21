// Round and match state machines. Plain structs, trivially copyable, so that
// search can clone them cheaply. See docs/RULES.md sections 7, 8 and 9.
#pragma once

#include <array>
#include <cstdint>

#include "gd/action.h"
#include "gd/cards.h"
#include "gd/config.h"

namespace gd {

enum class Phase : uint8_t {
  Deal = 0,
  Tribute = 1,
  BackTribute = 2,
  Play = 3,
  RoundEnd = 4,
  MatchEnd = 5,
};

// One public card movement of the tribute phase.
struct TributeMove {
  int8_t payer = -1;
  int8_t receiver = -1;
  CardId card = 0;
  bool back = false;    // false: tribute, true: back-tribute
};

struct RoundState {
  std::array<Hand, 4> hands{};
  std::array<Hand, 4> played{};      // cumulative, per seat, for the round
  int8_t level = 0;                  // round level, rank index 0..12
  int8_t to_move = 0;
  int8_t holder = -1;                // seat that played `top`, -1 when leading
  Action top{};                      // type Pass means the trick is open
  int8_t passes = 0;
  std::array<int8_t, 4> finish_pos{{-1, -1, -1, -1}};   // -1 while active
  std::array<int8_t, 4> order{{-1, -1, -1, -1}};        // seats by finish position
  int8_t num_finished = 0;
  Phase phase = Phase::Deal;
  int16_t steps = 0;
  // Most recent action of each seat this round, pass included. Type Pass with
  // no cards means the seat has not acted yet. Feeds the observation's
  // last-action block (DESIGN.md 6).
  std::array<Action, 4> last_action{};
  std::array<bool, 4> has_acted{{false, false, false, false}};

  // Tribute bookkeeping for this round.
  std::array<TributeMove, 4> tribute_moves{};
  int8_t num_tribute_moves = 0;
  bool anti_tribute = false;
  std::array<bool, 4> held_red_joker{{false, false, false, false}};  // at anti-tribute
  // Pending tribute state: payers still to act and their chosen cards.
  std::array<int8_t, 2> tribute_payers{{-1, -1}};
  std::array<int8_t, 2> tribute_receivers{{-1, -1}};
  std::array<int16_t, 2> tribute_cards{{-1, -1}};
  int8_t tribute_step = 0;           // index into the pending list
  int8_t first_leader = -1;

  bool active(int seat) const { return finish_pos[seat] < 0; }
  int num_active() const {
    return (active(0) ? 1 : 0) + (active(1) ? 1 : 0) + (active(2) ? 1 : 0) + (active(3) ? 1 : 0);
  }
  uint64_t hash() const;
};

struct MatchState {
  std::array<int8_t, 2> levels{{0, 0}};   // rank index per team
  std::array<int8_t, 2> fails{{0, 0}};    // cumulative failed level-A attempts
  int8_t owner = -1;                      // team whose level this round is played at
  int8_t round_index = 0;
  int8_t winner = -1;                     // team that passed A, -1 while running
  std::array<int8_t, 4> prev_order{{-1, -1, -1, -1}};
  bool has_prev = false;
  RoundState round{};
  uint64_t rng = 0;

  uint64_t hash() const;
};

// An explicit deal for duplicate evaluation and scenario tests.
// Either `leader` or `prev_order` decides how the round starts: with a leader
// the round begins in the play phase, with a previous order it begins with
// its tribute phase.
struct DealSpec {
  std::array<Hand, 4> hands{};
  int8_t level = 0;
  std::array<int8_t, 2> team_levels{{0, 0}};
  std::array<int8_t, 2> fails{{0, 0}};
  int8_t owner = -1;
  int8_t leader = -1;
  std::array<int8_t, 4> prev_order{{-1, -1, -1, -1}};
  bool has_prev = false;
};

struct RoundResult {
  std::array<int8_t, 4> order{{-1, -1, -1, -1}};
  int8_t winning_team = -1;
  int8_t gain = 0;
  std::array<int8_t, 2> levels{{0, 0}};
  std::array<int8_t, 2> fails{{0, 0}};
  int8_t next_owner = -1;
  int8_t match_winner = -1;
  int8_t round_level = 0;
  // Per-seat return used as the DMC target (DESIGN.md 8.2).
  std::array<int8_t, 4> seat_return{{0, 0, 0, 0}};
};

class Engine {
 public:
  explicit Engine(RuleConfig rules = RuleConfig::house(),
                  ActionConfig actions = ActionConfig{})
      : rules_(rules), actions_(actions) {}

  const RuleConfig& rules() const { return rules_; }
  const ActionConfig& action_config() const { return actions_; }

  // Start a match at level 2 for both teams. Deals from `rng`.
  void new_match(MatchState& m, uint64_t seed) const;
  // Start a match or round from an explicit deal.
  void set_deal(MatchState& m, const DealSpec& deal) const;
  // Deal and open the next round of a running match, applying tribute rules.
  void begin_round(MatchState& m) const;

  // Candidate actions for the seat to move in the current phase.
  void legal_actions(const MatchState& m, std::vector<Action>& out) const;
  // Apply one action by the seat to move. Advances phases and resolves forced
  // moves (a seat with pass as its only option passes without being asked).
  void apply(MatchState& m, const Action& a) const;
  // True when the current phase needs a decision from a network or a bot.
  bool needs_decision(const MatchState& m) const;
  // Close the round, fill `out` and update levels, ownership and the fail count.
  void end_round(MatchState& m, RoundResult& out) const;

 private:
  RuleConfig rules_;
  ActionConfig actions_;
};

// Oracle-equivalent match bookkeeping (gd_reference.end_of_round).
struct EndOfRound {
  std::array<int8_t, 2> levels{};
  std::array<int8_t, 2> fails{};
  int8_t next_owner = -1;
  int8_t match_winner = -1;
};
EndOfRound end_of_round(const std::array<int8_t, 2>& levels,
                        const std::array<int8_t, 2>& fails, int owner,
                        int round_level, const std::array<int8_t, 4>& order,
                        const RuleConfig& rules);

// Double tribute pairing (RULES.md 9.2). Returns the payer to the Banker, the
// payer to the Follower and the leader.
struct TributeRouting { int to_banker; int to_follower; int leader; };
TributeRouting double_tribute(int banker, int seat_a, int power_a, int seat_b,
                              int power_b, const RuleConfig& rules);

}  // namespace gd
