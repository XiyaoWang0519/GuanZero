// Vectorized environment. See docs/DESIGN.md section 5.3 and 8.1.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "gd/bots.h"
#include "gd/encoder.h"
#include "gd/state.h"

namespace gd {

struct EnvConfig {
  RuleConfig rules = RuleConfig::house();
  ActionConfig actions{};
  int history_len = 0;      // v2 only, unused in v1
  bool encode = true;       // fill obs and cand buffers
  bool log_public_actions = false;  // opt-in ordered events, forced passes included
  int log_env_limit = -1;   // log only env_id < limit; -1 logs all environments
};

// Flat buffers describing every environment that currently needs a decision.
// The spans alias the VecEnv's ordinary CPU storage and stay valid until the
// next pending(), step(), reset(), or fork(). CUDA callers pin their staging
// tensors explicitly; std::vector storage is not CUDA-pinned memory.
struct DecisionBatch {
  std::span<const float> obs;       // [rows, kObsDim]
  std::span<const float> cand;      // [offsets.back(), kActDim]
  std::span<const int32_t> offsets; // [rows + 1]
  std::span<const int32_t> env_id;  // [rows]
  std::span<const int32_t> seat;    // [rows]
  std::span<const int32_t> phase;   // [rows], Phase as int
  std::span<const int32_t> round_index;   // [rows], zero based within match
  std::span<const int64_t> match_id;      // [rows], per-environment generation
  std::span<const int32_t> greedy_choice; // [rows], local candidate index
  // Local candidate index of the styled bot under this environment's and seat's
  // style. Equal to greedy_choice while no styles are set.
  std::span<const int32_t> styled_choice; // [rows]
  // Privileged supervision only. Never concatenate these into policy input.
  std::span<const uint8_t> hidden_counts; // [rows, 3, 54], seats +1, +2, +3
  int rows = 0;
};

struct PublicActionEvent {
  int32_t env_id = -1;
  int64_t match_id = 0;
  int32_t round_index = 0;
  int32_t step = 0;           // play step before the action; tribute does not increment it
  int8_t seat = 0;
  Phase phase = Phase::Play;
  Action action{};
  std::array<float, kActDim> encoded_action{}; // tribute private flags always zeroed
  int32_t cards_left = 0;    // actor's public count after this action
  bool forced = false;
};

class VecEnv {
 public:
  VecEnv(int num_envs, int num_threads, EnvConfig cfg, uint64_t seed);
  ~VecEnv();
  VecEnv(const VecEnv&) = delete;
  VecEnv& operator=(const VecEnv&) = delete;

  // Empty `deals` means random deals from the seed.
  // Optional explicit match seeds preserve the serial evaluation seed schedule.
  // Mutually exclusive with deals; one seed per slot. Training omits these.
  void reset(std::span<const DealSpec> deals = {}, std::span<const uint64_t> match_seeds = {});
  // Advance every environment until it needs a decision or its match ends,
  // then describe the pending decisions.
  DecisionBatch pending();
  // One choice index per pending row, in the order pending() returned them.
  void step(std::span<const int32_t> choice_index);
  // Per-seat styles for styled_choice, laid out [num_envs, 4, StyleParams::kDim]
  // row-major. Copied into storage the environment owns. Forked slots inherit
  // the style rows of their source environment. While a batch is pending, both
  // calls recompute its styled_choice in place (the batch span and any view of
  // it see the new values), identical to what a fresh pending() would give, so
  // a caller may restyle an environment whose match restarted inside pending().
  void set_styles(std::span<const float> styles);
  void clear_styles();

  // Rounds finished since the last call.
  std::span<const RoundResult> drain_finished_rounds();
  // Optional public actions since the previous drain, ordered per environment.
  std::span<const PublicActionEvent> drain_public_actions();
  // Copy environment `env_id` at its current decision point into `copies`
  // fresh slots. Used for the counterfactual tribute branches of DESIGN.md 8.3
  // and later for endgame search. Returns the new environment ids.
  std::vector<int> fork(int env_id, int copies);

  int num_envs() const;
  const EnvConfig& config() const;
  // Candidate actions of a pending row, for the Python side and for tests.
  const std::vector<Action>& row_actions(int row) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gd
