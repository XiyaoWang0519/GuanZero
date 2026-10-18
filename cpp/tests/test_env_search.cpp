// VecEnv support for search rollouts: load() starts environments from given
// states, and stop_at_round_end leaves an environment idle once its round
// ends. A greedy VecEnv rollout from loaded states must replay a scalar Engine
// rollout of the same states exactly, round result included.
#include <cstdint>
#include <span>
#include <vector>

#include "gd/bots.h"
#include "gd/env.h"
#include "test_util.h"

namespace {

std::vector<gd::MatchState> fresh_states(int n, uint64_t seed) {
  gd::Engine engine;
  std::vector<gd::MatchState> out(n);
  for (int i = 0; i < n; ++i) engine.new_match(out[i], seed + uint64_t(i));
  return out;
}

// Scalar reference: lone passes are applied, every other decision takes the
// greedy bot under a copy of the state RNG, exactly as VecEnv reports it.
gd::RoundResult scalar_greedy_round(gd::MatchState m) {
  gd::Engine engine;
  engine.set_auto_pass(false);
  std::vector<gd::Action> cands;
  for (int guard = 0; guard < 4096; ++guard) {
    if (m.round.phase == gd::Phase::RoundEnd) {
      gd::RoundResult r;
      engine.end_round(m, r);
      return r;
    }
    cands.clear();
    engine.legal_actions(m, cands);
    if (cands.size() == 1 && cands[0].is_pass()) {
      engine.apply(m, cands[0]);
      continue;
    }
    uint64_t rng = m.rng;
    engine.apply(m, cands[gd::greedy_bot(m, cands, rng)]);
  }
  return {};
}

}  // namespace

TEST("vecenv: load + stop_at_round_end replays scalar greedy rounds and goes idle") {
  constexpr int kEnvs = 5;
  const auto states = fresh_states(kEnvs, 20261018);
  gd::EnvConfig cfg;
  cfg.stop_at_round_end = true;
  gd::VecEnv env(2, 1, cfg, 7);   // load() resizes the slots
  env.load(std::span<const gd::MatchState>(states.data(), states.size()));
  CHECK_EQ(env.num_envs(), kEnvs);
  std::vector<gd::RoundResult> results;
  for (int guard = 0; guard < 4096; ++guard) {
    const gd::DecisionBatch batch = env.pending();
    for (const auto& r : env.drain_finished_rounds()) results.push_back(r);
    if (batch.rows == 0) break;
    std::vector<int32_t> choice(batch.greedy_choice.begin(), batch.greedy_choice.end());
    env.step(choice);
  }
  CHECK_EQ(env.idle_count(), kEnvs);
  CHECK_EQ(static_cast<int>(results.size()), kEnvs);
  for (const auto& r : results) {
    const gd::RoundResult want = scalar_greedy_round(states[size_t(r.env_id)]);
    CHECK(r.order == want.order);
    CHECK_EQ(int(r.winning_team), int(want.winning_team));
    CHECK(r.seat_return == want.seat_return);
  }
  // Idle environments stay out of every later batch.
  CHECK_EQ(env.pending().rows, 0);
}

TEST("vecenv: load encodes the loaded state's first decision") {
  const auto states = fresh_states(3, 99);
  gd::VecEnv env(1, 1, gd::EnvConfig{}, 1);
  env.load(std::span<const gd::MatchState>(states.data(), states.size()));
  const gd::DecisionBatch batch = env.pending();
  CHECK_EQ(batch.rows, 3);
  for (int r = 0; r < batch.rows; ++r) {
    const gd::MatchState& m = states[size_t(batch.env_id[r])];
    std::vector<float> obs(gd::kObsDim);
    gd::encode_observation(m, m.round.to_move, std::span<float>(obs.data(), obs.size()));
    bool same = true;
    for (int k = 0; k < gd::kObsDim; ++k)
      same = same && obs[size_t(k)] == batch.obs[size_t(r) * gd::kObsDim + size_t(k)];
    CHECK(same);
  }
}

TEST("vecenv: without stop_at_round_end a loaded environment continues to the next round") {
  const auto states = fresh_states(1, 5);
  gd::VecEnv env(1, 1, gd::EnvConfig{}, 3);
  env.load(std::span<const gd::MatchState>(states.data(), states.size()));
  int rounds = 0;
  for (int guard = 0; guard < 4096 && rounds < 2; ++guard) {
    const gd::DecisionBatch batch = env.pending();
    rounds += static_cast<int>(env.drain_finished_rounds().size());
    std::vector<int32_t> choice(batch.greedy_choice.begin(), batch.greedy_choice.end());
    env.step(choice);
  }
  CHECK(rounds >= 2);
  CHECK_EQ(env.idle_count(), 0);
}
