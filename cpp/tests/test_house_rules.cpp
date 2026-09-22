// RULES.md O1 (double tribute with equal cards) and O4 (failures at level A)
// under both settings of each field: T-TRB-05, T-TRB-06, T-FLOW-10, T-FLOW-12
// and T-MATCH-04/06.
#include <vector>

#include "gd/cards.h"
#include "gd/config.h"
#include "gd/state.h"
#include "test_util.h"

using namespace gd;

namespace {

RuleConfig with_tie(TributeTie tie) {
  RuleConfig r = RuleConfig::house();
  r.tribute_tie = tie;
  return r;
}

RuleConfig with_fail_limit(int limit) {
  RuleConfig r = RuleConfig::house();
  r.a_fail_limit = limit;
  return r;
}

void apply_cards(const Engine& e, MatchState& m, const char* cards) {
  const Hand want = Hand::from_string(cards);
  std::vector<Action> actions;
  e.legal_actions(m, actions);
  for (const Action& a : actions) {
    if (a.cards == want) { e.apply(m, a); return; }
  }
  CHECK(false);  // the scripted move must be legal
}

// T-FLOW-10: previous double win by 0 and 2, seats 1 and 3 both give SB. The
// Banker returns S3 and the Follower S7, so where they land shows the pairing.
MatchState flow_10(const Engine& e) {
  DealSpec d;
  d.level = 12;
  d.team_levels = {12, 5};
  d.owner = 0;
  d.hands = {Hand::from_string("S3 D3 S4"), Hand::from_string("SB S6 D6"),
             Hand::from_string("S7 D7 S8"), Hand::from_string("SB S9 D9")};
  d.has_prev = true;
  d.prev_order = {0, 2, 1, 3};
  MatchState m;
  e.set_deal(m, d);
  CHECK(m.round.phase == Phase::Tribute);
  apply_cards(e, m, "SB");
  apply_cards(e, m, "SB");
  // Receivers return in payer order, so which one moves first depends on
  // the pairing.
  for (int i = 0; i < 2; ++i) {
    CHECK(m.round.phase == Phase::BackTribute);
    apply_cards(e, m, m.round.to_move == 0 ? "S3" : "S7");
  }
  CHECK(m.round.phase == Phase::Play);
  return m;
}

}  // namespace

TEST("house defaults follow the official rules: tie downstream, no A reset") {
  CHECK(RuleConfig::house().tribute_tie == TributeTie::Downstream);
  CHECK(RuleConfig{}.tribute_tie == TributeTie::Downstream);
  CHECK_EQ(RuleConfig::house().a_fail_limit, 0);
  // ogd keeps its own values for the parity tests.
  CHECK(RuleConfig::ogd().tribute_tie == TributeTie::LastFinisher);
  CHECK_EQ(RuleConfig::ogd().a_fail_limit, 4);
}

TEST("T-TRB-05 tie under downstream: (B+1) pays the Banker and leads") {
  const RuleConfig r = with_tie(TributeTie::Downstream);
  TributeRouting t = double_tribute(0, 1, 13, 3, 13, r);
  CHECK_EQ(t.to_banker, 1); CHECK_EQ(t.to_follower, 3); CHECK_EQ(t.leader, 1);
  t = double_tribute(0, 3, 13, 1, 13, r);  // payer order does not matter
  CHECK_EQ(t.to_banker, 1); CHECK_EQ(t.to_follower, 3); CHECK_EQ(t.leader, 1);
  t = double_tribute(3, 0, 12, 2, 12, r);  // wraps: Banker 3, downstream 0
  CHECK_EQ(t.to_banker, 0); CHECK_EQ(t.to_follower, 2); CHECK_EQ(t.leader, 0);
}

TEST("T-TRB-05 tie under upstream: (B+3) pays the Banker and leads") {
  const RuleConfig r = with_tie(TributeTie::Upstream);
  TributeRouting t = double_tribute(0, 1, 13, 3, 13, r);
  CHECK_EQ(t.to_banker, 3); CHECK_EQ(t.to_follower, 1); CHECK_EQ(t.leader, 3);
  t = double_tribute(1, 2, 12, 0, 12, r);
  CHECK_EQ(t.to_banker, 0); CHECK_EQ(t.to_follower, 2); CHECK_EQ(t.leader, 0);
}

TEST("T-TRB-06 the higher card decides under every tie mode") {
  for (TributeTie tie : {TributeTie::Downstream, TributeTie::Upstream}) {
    const RuleConfig r = with_tie(tie);
    TributeRouting t = double_tribute(0, 1, 14, 3, 13, r);
    CHECK_EQ(t.to_banker, 1); CHECK_EQ(t.to_follower, 3); CHECK_EQ(t.leader, 1);
    t = double_tribute(0, 1, 11, 3, 12, r);
    CHECK_EQ(t.to_banker, 3); CHECK_EQ(t.to_follower, 1); CHECK_EQ(t.leader, 3);
  }
}

TEST("T-FLOW-10 house: clockwise tribute, seat 1 pays 0 and leads") {
  const Engine e;  // house default
  const MatchState m = flow_10(e);
  CHECK_EQ(m.round.hands[1].count(static_cast<CardId>(card_from_string("S3"))), 1);
  CHECK_EQ(m.round.hands[3].count(static_cast<CardId>(card_from_string("S7"))), 1);
  CHECK_EQ(static_cast<int>(m.round.to_move), 1);
}

TEST("T-FLOW-10 upstream: seat 3 pays 0 and leads") {
  const Engine e(with_tie(TributeTie::Upstream));
  const MatchState m = flow_10(e);
  CHECK_EQ(m.round.hands[3].count(static_cast<CardId>(card_from_string("S3"))), 1);
  CHECK_EQ(m.round.hands[1].count(static_cast<CardId>(card_from_string("S7"))), 1);
  CHECK_EQ(static_cast<int>(m.round.to_move), 3);
}

TEST("T-FLOW-12 and T-MATCH-04/06 house: no reset after the third failure") {
  const RuleConfig r = RuleConfig::house();
  // T-MATCH-04: third failure by losing the round
  EndOfRound e = end_of_round({12, 5}, {2, 0}, 0, 12, {1, 3, 0, 2}, r);
  CHECK_EQ(e.levels[0], 12); CHECK_EQ(e.levels[1], 8);
  CHECK_EQ(e.fails[0], 3); CHECK_EQ(e.fails[1], 0);
  CHECK_EQ(e.next_owner, 1); CHECK_EQ(e.match_winner, -1);
  // T-MATCH-06: third failure while winning with Banker plus Dweller
  e = end_of_round({12, 5}, {2, 0}, 0, 12, {0, 1, 3, 2}, r);
  CHECK_EQ(e.levels[0], 12); CHECK_EQ(e.levels[1], 5);
  CHECK_EQ(e.fails[0], 3); CHECK_EQ(e.next_owner, 0); CHECK_EQ(e.match_winner, -1);
}

TEST("T-FLOW-12 and T-MATCH-04/06 with a_fail_limit 3: reset to the deuce") {
  const RuleConfig r = with_fail_limit(3);
  EndOfRound e = end_of_round({12, 5}, {2, 0}, 0, 12, {1, 3, 0, 2}, r);
  CHECK_EQ(e.levels[0], 0); CHECK_EQ(e.levels[1], 8);
  CHECK_EQ(e.fails[0], 0); CHECK_EQ(e.next_owner, 1);
  e = end_of_round({12, 5}, {2, 0}, 0, 12, {0, 1, 3, 2}, r);
  CHECK_EQ(e.levels[0], 0); CHECK_EQ(e.levels[1], 5);
  CHECK_EQ(e.fails[0], 0); CHECK_EQ(e.next_owner, 0);
}
