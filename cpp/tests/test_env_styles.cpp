// VecEnv style rows and the pending batch (STAGE_B_TODO B7). A match can end
// and the next one open inside a single pending(), so an opponent source only
// learns of the new match after the batch exists. set_styles and clear_styles
// must therefore recompute styled_choice of the rows already pending, in the
// storage the batch spans alias, exactly as a fresh pending() would.
#include <cstdint>
#include <span>
#include <vector>

#include "gd/bots.h"
#include "gd/env.h"
#include "test_util.h"

namespace {

constexpr int kEnvs = 8;
constexpr uint64_t kSeed = 20260922;

gd::EnvConfig quiet() {
  gd::EnvConfig cfg;
  cfg.encode = false;
  return cfg;
}

// Sampling styles that differ by environment and seat, so a stale row shows.
std::vector<float> hot_styles(int envs) {
  std::vector<float> out;
  out.reserve(size_t(envs) * 4 * gd::StyleParams::kDim);
  for (int e = 0; e < envs; ++e)
    for (int seat = 0; seat < 4; ++seat) {
      gd::StyleParams p = gd::StyleParams::neutral();
      p.v[gd::StyleParams::kBombThreshold] = 0.0f;
      p.v[gd::StyleParams::kLeadHighBias] = seat % 2 ? 1.0f : -1.0f;
      p.v[gd::StyleParams::kTypePref + 1] = 0.25f * float(e % 4);
      p.v[gd::StyleParams::kTemperature] = 1.0f;
      out.insert(out.end(), p.v.begin(), p.v.end());
    }
  return out;
}

std::vector<float> neutral_styles(int envs) {
  std::vector<float> out;
  const auto n = gd::StyleParams::neutral();
  for (int k = 0; k < envs * 4; ++k) out.insert(out.end(), n.v.begin(), n.v.end());
  return out;
}

}  // namespace

TEST("vecenv: set_styles after a match restart equals a fresh pending under those styles") {
  gd::VecEnv live(kEnvs, 2, quiet(), kSeed);
  gd::VecEnv fresh(kEnvs, 2, quiet(), kSeed);
  const std::vector<float> before = neutral_styles(kEnvs);
  const std::vector<float> after = hot_styles(kEnvs);
  live.reset();
  fresh.reset();
  live.set_styles(before);
  fresh.set_styles(after);
  std::vector<int64_t> last_match(kEnvs, 0);
  int restarts = 0, changed = 0;
  for (int step = 0; step < 200000 && restarts < 12; ++step) {
    gd::DecisionBatch a = live.pending();
    gd::DecisionBatch b = fresh.pending();
    CHECK_EQ(a.rows, b.rows);
    bool restart = false;
    for (int r = 0; r < a.rows; ++r) {
      const int e = a.env_id[r];
      if (a.match_id[r] != last_match[e]) { restart = true; ++restarts; }
      last_match[e] = a.match_id[r];
    }
    if (restart) {
      const std::vector<int32_t> stale(a.styled_choice.begin(), a.styled_choice.end());
      live.set_styles(after);
      // Read through the span returned before set_styles: same storage.
      for (int r = 0; r < a.rows; ++r) {
        CHECK_EQ(a.styled_choice[r], b.styled_choice[r]);
        changed += a.styled_choice[r] != stale[r];
      }
      live.clear_styles();
      for (int r = 0; r < a.rows; ++r) CHECK_EQ(a.styled_choice[r], a.greedy_choice[r]);
      live.set_styles(before);
    }
    // Both follow greedy, which no style row influences, so they stay in step.
    std::vector<int32_t> choice(a.greedy_choice.begin(), a.greedy_choice.end());
    live.step(choice);
    fresh.step(choice);
    live.drain_finished_rounds();
    fresh.drain_finished_rounds();
  }
  CHECK(restarts >= 12);
  CHECK(changed > 0);  // the refresh must be observable, or the test is vacuous
}

TEST("vecenv: set_styles between step and pending changes nothing pending") {
  gd::VecEnv env(4, 1, quiet(), kSeed + 1);
  env.reset();
  gd::DecisionBatch batch = env.pending();
  std::vector<int32_t> choice(batch.greedy_choice.begin(), batch.greedy_choice.end());
  env.step(choice);
  env.set_styles(hot_styles(4));  // no batch pending: must not touch stale rows
  env.clear_styles();
  gd::DecisionBatch next = env.pending();
  for (int r = 0; r < next.rows; ++r) CHECK_EQ(next.styled_choice[r], next.greedy_choice[r]);
}
