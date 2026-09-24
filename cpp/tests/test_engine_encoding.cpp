#include "test_util.h"

#include <algorithm>
#include <array>
#include <random>

#include "gd/encoder.h"
#include "gd/movegen.h"

TEST("HandView preserves every rank suit and multiplicity at every level") {
  std::mt19937 rng(71029);
  for (int trial = 0; trial < 200; ++trial) {
    gd::Hand hand;
    std::array<int, gd::kNumCardIds> counts{};
    for (int c = 0; c < gd::kNumCardIds; ++c) {
      // Include empty/full decks as well as sparse and dense multisets.
      counts[c] = trial == 0 ? 0 : trial == 1 ? 2 : int(rng() % 3);
      hand.add_n(static_cast<gd::CardId>(c), counts[c]);
    }
    for (int level = 0; level < gd::kNumRanks; ++level) {
      const auto view = gd::make_view(hand, level);
      std::array<int, gd::kNumPowers> ranks{};
      std::array<uint16_t, gd::kNumSuits> suits{};
      int size = 0;
      for (int c = 0; c < gd::kNumCardIds; ++c) {
        CHECK_EQ(view.card_count[c], counts[c]);
        size += counts[c];
        if (c != gd::wild_id(level)) ranks[gd::rank_of(c)] += counts[c];
        if (c < 52 && counts[c]) suits[gd::suit_of(c)] |= 1u << gd::rank_of(c);
      }
      CHECK_EQ(view.size, size);
      CHECK_EQ(view.wilds, counts[gd::wild_id(level)]);
      for (int rank = 0; rank < gd::kNumPowers; ++rank)
        CHECK_EQ(view.rank_count[rank], ranks[rank]);
      CHECK(view.suit_rank_mask == suits);
    }
  }
}

TEST("unseen card encoding preserves both copies across observers and phases") {
  gd::Engine engine;
  gd::MatchState state;
  engine.new_match(state, 24681);
  std::mt19937 rng(97531);
  std::vector<gd::Action> actions;
  std::array<float, gd::kObsDim> obs;
  for (int step = 0; step < 1000; ++step) {
    if (state.round.phase == gd::Phase::RoundEnd) {
      gd::RoundResult result;
      engine.end_round(state, result);
      if (state.winner >= 0) engine.new_match(state, rng());
      else engine.begin_round(state);
    }
    for (int observer = 0; observer < 4; ++observer) {
      obs.fill(-1.0f);  // Encoding must also clear values from the previous row.
      gd::encode_observation(state, observer, obs);
      for (int c = 0; c < gd::kNumCardIds; ++c) {
        int unseen = 2 - state.round.hands[observer].count(c);
        for (const auto& played : state.round.played) unseen -= played.count(c);
        CHECK_EQ(obs[gd::kObsUnseen + c], unseen >= 1 ? 1.0f : 0.0f);
        CHECK_EQ(obs[gd::kObsUnseen + gd::kNumCardIds + c], unseen == 2 ? 1.0f : 0.0f);
      }
    }
    actions.clear();
    engine.legal_actions(state, actions);
    CHECK(!actions.empty());
    if (actions.empty()) return;
    engine.apply(state, actions[rng() % actions.size()]);
  }
}

TEST("move generation appends while preserving existing actions") {
  const auto hand = gd::Hand::from_string("H7 H7 S9 D9 C9 S3 D3 SA D2 C3 S4 H5 SB SB");
  gd::Action top;
  top.type = gd::Type::Single;
  top.key = 3;
  gd::Action sentinel;
  sentinel.type = gd::Type::BackTribute;
  sentinel.key = 9;
  sentinel.cards = gd::Hand::from_string("DA");
  for (const auto& config : {gd::ActionConfig{}, gd::ActionConfig::full()}) {
    for (const auto& trick : {gd::Action{}, top}) {
      std::vector<gd::Action> expected;
      gd::generate_moves(hand, 5, trick, config, gd::RuleConfig::house(), expected);
      std::vector<gd::Action> appended{sentinel, sentinel};
      gd::generate_moves(hand, 5, trick, config, gd::RuleConfig::house(), appended);
      CHECK_EQ(appended.size(), expected.size() + 2);
      CHECK(appended[0] == sentinel && appended[1] == sentinel);
      CHECK(std::equal(expected.begin(), expected.end(), appended.begin() + 2));
    }
  }
}
