#include "test_util.h"

#include <array>
#include <type_traits>

#include "gd/encoder.h"
#include "gd/state.h"

namespace {

gd::MatchState tribute_state(const gd::Engine& engine, const char* receiver_hand) {
  gd::DealSpec deal;
  deal.level = 12;
  deal.hands = {gd::Hand::from_string(receiver_hand), gd::Hand::from_string("D4"),
                gd::Hand::from_string("C5"), gd::Hand::from_string("S4 S9")};
  deal.has_prev = true;
  deal.prev_order = {0, 1, 2, 3};
  gd::MatchState state;
  engine.set_deal(state, deal);
  return state;
}

void apply_card(const gd::Engine& engine, gd::MatchState& state, const char* card) {
  const gd::Hand wanted = gd::Hand::from_string(card);
  std::vector<gd::Action> actions;
  engine.legal_actions(state, actions);
  for (const auto& action : actions) {
    if (action.cards.has1 == wanted.has1 && action.cards.has2 == wanted.has2) {
      engine.apply(state, action);
      return;
    }
  }
  CHECK(false);  // The scripted transfer/play must be legal in the real engine.
}

float known_card(const gd::MatchState& state, int observer, int holder, const char* card) {
  const int rel = (holder - observer + 4) % 4;
  CHECK(rel > 0);
  std::array<float, gd::kObsDim> obs{};
  gd::encode_observation(state, observer, obs);
  return obs[gd::kObsKnownHoldings + (rel - 1) * 54 + gd::card_from_string(card)];
}

}  // namespace

TEST("long match round indices do not wrap after 127") {
  static_assert(std::is_trivially_copyable_v<gd::MatchState>);
  gd::Engine engine;
  gd::MatchState state;
  engine.new_match(state, 41);
  state.round_index = 127;
  engine.begin_round(state);
  CHECK_EQ(state.round_index, 128);
  state.round.order = {0, 2, 1, 3};
  state.round.num_finished = 4;
  state.round.num_out = 2;
  state.round.phase = gd::Phase::RoundEnd;
  gd::RoundResult result;
  engine.end_round(state, result);
  CHECK_EQ(result.round_index, 128);
  CHECK_EQ(result.env_id, -1);
}

TEST("public last action cannot expose private tribute structure") {
  gd::Engine engine;
  gd::MatchState a;
  engine.new_match(a, 10);
  const int actor = (a.round.to_move + 1) % 4;
  a.round.has_acted[actor] = true;
  a.round.last_action[actor].type = gd::Type::BackTribute;
  a.round.last_action[actor].cards.add(gd::card_of(3, 0));
  gd::MatchState b = a;
  // Only private card allocation changes; retain each public hand size.
  std::swap(b.round.hands[actor], b.round.hands[(actor + 1) % 4]);
  std::array<float, gd::kObsDim> oa{}, ob{};
  gd::encode_observation(a, a.round.to_move, oa);
  gd::encode_observation(b, b.round.to_move, ob);
  CHECK(oa == ob);
  for (int i = gd::kActTributeFlags; i < gd::kActDim; ++i)
    CHECK_EQ(oa[gd::kObsLastAction + i], 0.0f);
}

TEST("T-OBS-01 returned tribute is known only at its final public holder") {
  gd::Engine engine;
  auto state = tribute_state(engine, "S3 SJ");
  CHECK(state.round.phase == gd::Phase::Tribute);
  apply_card(engine, state, "S9");
  CHECK(state.round.phase == gd::Phase::BackTribute);
  CHECK_EQ(known_card(state, 1, 0, "S9"), 1.0f);
  CHECK_EQ(known_card(state, 1, 3, "S9"), 0.0f);

  apply_card(engine, state, "S9");
  CHECK(state.round.phase == gd::Phase::Play);
  CHECK_EQ(known_card(state, 1, 0, "S9"), 0.0f);
  CHECK_EQ(known_card(state, 1, 3, "S9"), 1.0f);
}

TEST("T-OBS-02 returning one of two copies does not reveal the private copy") {
  gd::Engine engine;
  auto state = tribute_state(engine, "S9 SJ");
  const auto nine = gd::card_from_string("S9");
  apply_card(engine, state, "S9");
  CHECK_EQ(state.round.hands[0].count(nine), 2);
  CHECK_EQ(known_card(state, 1, 0, "S9"), 1.0f);

  apply_card(engine, state, "S9");
  CHECK_EQ(state.round.hands[0].count(nine), 1);
  CHECK_EQ(known_card(state, 1, 0, "S9"), 0.0f);
  CHECK_EQ(known_card(state, 1, 3, "S9"), 1.0f);
}

TEST("T-OBS-03 different back tribute preserves known cards until played") {
  gd::Engine engine;
  auto state = tribute_state(engine, "S3 SJ");
  apply_card(engine, state, "S9");
  apply_card(engine, state, "S3");
  CHECK_EQ(known_card(state, 1, 0, "S9"), 1.0f);
  CHECK_EQ(known_card(state, 1, 3, "S3"), 1.0f);

  apply_card(engine, state, "S4");
  CHECK_EQ(known_card(state, 1, 3, "S3"), 1.0f);
  apply_card(engine, state, "S9");
  CHECK_EQ(known_card(state, 1, 0, "S9"), 0.0f);
  CHECK_EQ(known_card(state, 1, 3, "S3"), 1.0f);
}

TEST("T-OBS-04 playing a known copy does not reveal a remaining private copy") {
  gd::Engine engine;
  auto state = tribute_state(engine, "S3 S9 SJ");
  const auto nine = gd::card_from_string("S9");
  apply_card(engine, state, "S9");
  apply_card(engine, state, "S3");
  CHECK_EQ(state.round.hands[0].count(nine), 2);
  CHECK_EQ(known_card(state, 1, 0, "S9"), 1.0f);

  apply_card(engine, state, "S4");
  apply_card(engine, state, "S9");
  CHECK_EQ(state.round.hands[0].count(nine), 1);
  CHECK_EQ(known_card(state, 1, 0, "S9"), 0.0f);
}
