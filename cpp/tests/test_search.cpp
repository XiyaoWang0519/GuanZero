#include "test_util.h"

#include <array>
#include <vector>

#include "gd/bots.h"
#include "gd/search.h"

namespace {

void check_conservation(const gd::MatchState& state) {
  for (int c = 0; c < gd::kNumCardIds; ++c) {
    int count = 0;
    for (int seat = 0; seat < 4; ++seat) {
      count += state.round.hands[seat].count(static_cast<gd::CardId>(c));
      count += state.round.played[seat].count(static_cast<gd::CardId>(c));
    }
    CHECK_EQ(count, 2);
  }
}

}  // namespace

TEST("uniform determinization conserves public counts and ignores true hidden cards") {
  gd::Engine engine;
  gd::MatchState state;
  engine.new_match(state, 123);
  gd::MatchState changed = state;
  std::swap(changed.round.hands[1], changed.round.hands[2]);
  const auto first = gd::determinize_uniform(state, 0, 876);
  const auto second = gd::determinize_uniform(changed, 0, 876);
  CHECK(first.round.hands == second.round.hands);
  CHECK(first.round.hands[0] == state.round.hands[0]);
  CHECK(state.round.hands[1] != first.round.hands[1] ||
        state.round.hands[2] != first.round.hands[2]);
  check_conservation(first);
  for (int seat = 0; seat < 4; ++seat)
    CHECK_EQ(first.round.hands[seat].size(), state.round.hands[seat].size());
}

TEST("uniform determinization retains publicly known tribute card") {
  gd::Engine engine;
  gd::MatchState state;
  engine.new_match(state, 456);
  const auto card = state.round.hands[1].to_vector().front();
  state.round.num_tribute_moves = 1;
  state.round.tribute_moves[0] = {3, 1, card, false};
  for (int seed = 0; seed < 30; ++seed) {
    const auto sample = gd::determinize_uniform(state, 0, seed);
    CHECK(sample.round.hands[1].count(card) >= 1);
    check_conservation(sample);
  }
}

TEST("uniform determinization respects anti-tribute red joker holders") {
  gd::Engine engine;
  gd::MatchState state;
  engine.new_match(state, 457);
  state.round.anti_tribute = true;
  for (int seat = 0; seat < 4; ++seat)
    state.round.held_red_joker[seat] = state.round.hands[seat].count(gd::kRJ) > 0;
  gd::MatchState same_information = state;
  bool swapped = false;
  for (int a = 1; a < 4 && !swapped; ++a)
    for (int b = a + 1; b < 4; ++b)
      if (state.round.hands[a].count(gd::kRJ) == state.round.hands[b].count(gd::kRJ)) {
        std::swap(same_information.round.hands[a], same_information.round.hands[b]);
        swapped = true;
        break;
      }
  CHECK(swapped);
  for (int seed = 0; seed < 30; ++seed) {
    const auto sample = gd::determinize_uniform(state, 0, seed);
    const auto alternate = gd::determinize_uniform(same_information, 0, seed);
    CHECK(sample.round.hands == alternate.round.hands);
    for (int seat = 0; seat < 4; ++seat)
      CHECK_EQ(sample.round.hands[seat].count(gd::kRJ),
               state.round.hands[seat].count(gd::kRJ));
    check_conservation(sample);
  }
}

TEST("non-anti private red-joker flags cannot change determinization") {
  gd::Engine engine;
  gd::MatchState state;
  engine.new_match(state, 458);
  gd::MatchState changed = state;
  changed.round.held_red_joker = {true, true, true, true};
  CHECK(!state.round.anti_tribute);
  for (int seed = 0; seed < 30; ++seed)
    CHECK(gd::determinize_uniform(state, 0, seed).round.hands ==
          gd::determinize_uniform(changed, 0, seed).round.hands);
}

TEST("uniform determinized clone replays the same actions") {
  gd::Engine engine;
  gd::MatchState state;
  engine.new_match(state, 789);
  uint64_t rng = 1;
  for (int i = 0; i < 10 && state.round.phase == gd::Phase::Play; ++i) {
    std::vector<gd::Action> actions;
    engine.legal_actions(state, actions);
    engine.apply(state, actions[gd::greedy_bot(state, actions, rng)]);
  }
  CHECK(state.round.phase == gd::Phase::Play);
  auto a = gd::determinize_uniform(state, state.round.to_move, 50);
  auto b = a;
  for (int i = 0; i < 100 && a.round.phase == gd::Phase::Play; ++i) {
    std::vector<gd::Action> actions;
    engine.legal_actions(a, actions);
    const auto action = actions.front();
    engine.apply(a, action);
    engine.apply(b, action);
    CHECK_EQ(a.hash(), b.hash());
  }
}
