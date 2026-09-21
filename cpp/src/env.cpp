// Vectorized environment. See docs/DESIGN.md sections 5.3 and 8.1.
#include "gd/env.h"

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#include "gd/movegen.h"

namespace gd {
namespace {

uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// A persistent pool, because the rollout loop calls into it thousands of times
// a second and spawning threads per call would dominate the cost.
class Pool {
 public:
  explicit Pool(int threads) {
    if (threads <= 1) return;
    workers_.reserve(threads);
    for (int i = 0; i < threads; ++i) workers_.emplace_back([this, i] { loop(i); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  int size() const { return static_cast<int>(workers_.size()); }

  // Runs body(shard) once per worker, or inline when there are no workers.
  void run(const std::function<void(int, int)>& body) {
    const int n = size();
    if (n == 0) { body(0, 1); return; }
    {
      std::lock_guard<std::mutex> lk(m_);
      body_ = &body;
      remaining_ = n;
      ++epoch_;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lk(m_);
    done_.wait(lk, [this] { return remaining_ == 0; });
    body_ = nullptr;
  }

 private:
  void loop(int index) {
    uint64_t seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&] { return stop_ || epoch_ != seen; });
      if (stop_) return;
      seen = epoch_;
      const std::function<void(int, int)>* body = body_;
      const int n = size();
      lk.unlock();
      if (body) (*body)(index, n);
      lk.lock();
      if (--remaining_ == 0) done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, done_;
  const std::function<void(int, int)>* body_ = nullptr;
  uint64_t epoch_ = 0;
  int remaining_ = 0;
  bool stop_ = false;
};

}  // namespace

struct VecEnv::Impl {
  EnvConfig cfg;
  Engine engine;
  Pool pool;
  uint64_t seed = 0;

  std::vector<MatchState> states;
  std::vector<uint64_t> rngs;
  std::vector<std::vector<Action>> cands;   // per environment, reused
  std::vector<uint8_t> waiting;             // per environment: needs a decision

  // Flat batch buffers, reused across iterations.
  std::vector<float> obs, cand;
  std::vector<int32_t> offsets, env_id, seat, phase;
  std::vector<int> rows_env;                // environment index per row
  std::vector<RoundResult> finished;
  std::vector<RoundResult> drained;
  std::vector<std::vector<RoundResult>> finished_shards;

  Impl(int num_envs, int num_threads, EnvConfig c, uint64_t s)
      : cfg(c), engine(c.rules, c.actions), pool(num_threads), seed(s) {
    states.resize(num_envs);
    rngs.resize(num_envs);
    cands.resize(num_envs);
    waiting.assign(num_envs, 0);
    finished_shards.resize(std::max(1, pool.size()));
  }

  // Advance one environment until it needs a decision. Round ends are drained
  // into the shard's own buffer so that threads never touch shared state.
  void advance(int i, std::vector<RoundResult>& out) {
    MatchState& m = states[i];
    for (int guard = 0; guard < 4096; ++guard) {
      if (m.round.phase == Phase::RoundEnd) {
        RoundResult r;
        engine.end_round(m, r);
        out.push_back(r);
        if (m.winner >= 0) {
          engine.new_match(m, splitmix64(rngs[i]));
        } else {
          engine.begin_round(m);
        }
        continue;
      }
      cands[i].clear();
      engine.legal_actions(m, cands[i]);
      if (cands[i].empty()) {           // defensive: restart a wedged match
        engine.new_match(m, splitmix64(rngs[i]));
        continue;
      }
      waiting[i] = 1;
      return;
    }
    waiting[i] = 0;
  }
};

VecEnv::VecEnv(int num_envs, int num_threads, EnvConfig cfg, uint64_t seed)
    : impl_(std::make_unique<Impl>(num_envs, num_threads, cfg, seed)) {}
VecEnv::~VecEnv() = default;

void VecEnv::reset(std::span<const DealSpec> deals) {
  Impl& s = *impl_;
  uint64_t root = s.seed;
  for (size_t i = 0; i < s.states.size(); ++i) {
    s.rngs[i] = splitmix64(root) | 1ULL;
    if (!deals.empty()) {
      s.engine.set_deal(s.states[i], deals[i % deals.size()]);
    } else {
      s.engine.new_match(s.states[i], splitmix64(s.rngs[i]));
    }
    s.waiting[i] = 0;
  }
  s.finished.clear();
}

DecisionBatch VecEnv::pending() {
  Impl& s = *impl_;
  const int n = static_cast<int>(s.states.size());
  for (auto& shard : s.finished_shards) shard.clear();

  s.pool.run([&](int shard, int shards) {
    auto& out = s.finished_shards[shard];
    for (int i = shard; i < n; i += shards)
      if (!s.waiting[i]) s.advance(i, out);
  });
  for (auto& shard : s.finished_shards)
    s.finished.insert(s.finished.end(), shard.begin(), shard.end());

  // Lay the rows out in environment order so that the batch is deterministic
  // however the work was sharded.
  s.rows_env.clear();
  s.offsets.clear();
  s.offsets.push_back(0);
  for (int i = 0; i < n; ++i) {
    if (!s.waiting[i]) continue;
    s.rows_env.push_back(i);
    s.offsets.push_back(s.offsets.back() + static_cast<int32_t>(s.cands[i].size()));
  }
  const int rows = static_cast<int>(s.rows_env.size());

  s.env_id.resize(rows);
  s.seat.resize(rows);
  s.phase.resize(rows);
  if (s.cfg.encode) {
    s.obs.assign(size_t(rows) * kObsDim, 0.0f);
    s.cand.assign(size_t(s.offsets.back()) * kActDim, 0.0f);
  } else {
    s.obs.clear();
    s.cand.clear();
  }

  s.pool.run([&](int shard, int shards) {
    for (int r = shard; r < rows; r += shards) {
      const int i = s.rows_env[r];
      const MatchState& m = s.states[i];
      s.env_id[r] = i;
      s.seat[r] = m.round.to_move;
      s.phase[r] = static_cast<int32_t>(m.round.phase);
      if (!s.cfg.encode) continue;
      encode_observation(m, m.round.to_move,
                         std::span<float>(s.obs.data() + size_t(r) * kObsDim, kObsDim));
      const int base = s.offsets[r];
      for (size_t k = 0; k < s.cands[i].size(); ++k)
        encode_action(s.cands[i][k], m.round, m.round.to_move,
                      std::span<float>(s.cand.data() + size_t(base + k) * kActDim, kActDim));
    }
  });

  DecisionBatch b;
  b.rows = rows;
  b.obs = s.obs;
  b.cand = s.cand;
  b.offsets = s.offsets;
  b.env_id = s.env_id;
  b.seat = s.seat;
  b.phase = s.phase;
  return b;
}

void VecEnv::step(std::span<const int32_t> choice_index) {
  Impl& s = *impl_;
  const int rows = static_cast<int>(s.rows_env.size());
  assert(static_cast<int>(choice_index.size()) == rows);
  s.pool.run([&](int shard, int shards) {
    for (int r = shard; r < rows; r += shards) {
      const int i = s.rows_env[r];
      const auto& list = s.cands[i];
      if (list.empty()) continue;
      int k = choice_index[r];
      if (k < 0 || k >= static_cast<int>(list.size())) k = 0;
      s.engine.apply(s.states[i], list[k]);
      s.waiting[i] = 0;
    }
  });
}

std::span<const RoundResult> VecEnv::drain_finished_rounds() {
  Impl& s = *impl_;
  s.drained.swap(s.finished);
  s.finished.clear();
  return s.drained;
}

std::vector<int> VecEnv::fork(int env_id, int copies) {
  Impl& s = *impl_;
  std::vector<int> ids;
  ids.reserve(copies);
  for (int c = 0; c < copies; ++c) {
    const int id = static_cast<int>(s.states.size());
    s.states.push_back(s.states[env_id]);
    s.rngs.push_back(splitmix64(s.rngs[env_id]));
    s.cands.push_back(s.cands[env_id]);
    s.waiting.push_back(s.waiting[env_id]);
    ids.push_back(id);
  }
  return ids;
}

int VecEnv::num_envs() const { return static_cast<int>(impl_->states.size()); }
const EnvConfig& VecEnv::config() const { return impl_->cfg; }
const std::vector<Action>& VecEnv::row_actions(int row) const {
  return impl_->cands[impl_->rows_env[row]];
}

}  // namespace gd
