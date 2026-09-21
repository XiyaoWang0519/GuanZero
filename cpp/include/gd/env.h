// Vectorized environment. See docs/DESIGN.md section 5.3 and 8.1.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "gd/encoder.h"
#include "gd/state.h"

namespace gd {

struct EnvConfig {
  RuleConfig rules = RuleConfig::house();
  ActionConfig actions{};
  int history_len = 0;      // v2 only, unused in v1
  bool encode = true;       // fill obs and cand buffers
};

// Flat buffers describing every environment that currently needs a decision.
// The spans alias the VecEnv's own pinned storage and stay valid until the next
// call to pending() or step().
struct DecisionBatch {
  std::span<const float> obs;       // [rows, kObsDim]
  std::span<const float> cand;      // [offsets.back(), kActDim]
  std::span<const int32_t> offsets; // [rows + 1]
  std::span<const int32_t> env_id;  // [rows]
  std::span<const int32_t> seat;    // [rows]
  std::span<const int32_t> phase;   // [rows], Phase as int
  int rows = 0;
};

class VecEnv {
 public:
  VecEnv(int num_envs, int num_threads, EnvConfig cfg, uint64_t seed);
  ~VecEnv();
  VecEnv(const VecEnv&) = delete;
  VecEnv& operator=(const VecEnv&) = delete;

  // Empty `deals` means random deals from the seed.
  void reset(std::span<const DealSpec> deals = {});
  // Advance every environment until it needs a decision or its match ends,
  // then describe the pending decisions.
  DecisionBatch pending();
  // One choice index per pending row, in the order pending() returned them.
  void step(std::span<const int32_t> choice_index);
  // Rounds finished since the last call.
  std::span<const RoundResult> drain_finished_rounds();
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
