// Vectorized environment. See docs/DESIGN.md sections 5.3 and 8.1.
#include "gd/env.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <thread>

#include "gd/movegen.h"
#include "gd/bots.h"

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
  std::vector<int64_t> match_ids;
  std::vector<std::vector<Action>> cands;   // per environment, reused
  std::vector<uint8_t> waiting;             // per environment: needs a decision

  // Flat batch buffers, reused across iterations.
  std::vector<float> obs, cand;
  std::vector<int32_t> offsets, env_id, seat, phase, round_index, greedy_choice;
  std::vector<int32_t> styled_choice;
  std::vector<float> styles;           // [envs, 4, StyleParams::kDim] or empty
  std::vector<uint64_t> style_rng_base;  // per environment, derived from seed
  std::vector<int64_t> batch_match_id;
  std::vector<uint8_t> hidden_counts;
  std::vector<int> rows_env;                // environment index per row
  std::vector<RoundResult> finished;
  std::vector<RoundResult> drained;
  std::vector<std::vector<RoundResult>> finished_shards;
  std::vector<std::vector<PublicActionEvent>> events; // per env, no thread contention
  std::vector<PublicActionEvent> drained_events;
  bool initialized = false;
  bool batch_ready = false;

  Impl(int num_envs, int num_threads, EnvConfig c, uint64_t s)
      : cfg(c), engine(c.rules, c.actions), pool(num_threads), seed(s) {
    states.resize(num_envs);
    rngs.resize(num_envs);
    match_ids.assign(num_envs, 0);
    cands.resize(num_envs);
    waiting.assign(num_envs, 0);
    style_rng_base.resize(num_envs);
    uint64_t root = s ^ 0xB0B5EED5B0B5EED5ULL;
    for (int i = 0; i < num_envs; ++i) style_rng_base[i] = splitmix64(root) | 1ULL;
    finished_shards.resize(std::max(1, pool.size()));
    events.resize(num_envs);
    // advance() resolves forced passes from the candidate list it generates
    // anyway. With auto-pass on, Engine::apply would generate the next seat's
    // moves once more only to detect them.
    engine.set_auto_pass(false);
  }

  // Styled-bot choice of pending row `r` under the current style rows. Needs
  // greedy_choice[r] and rows_env to be filled for the current batch.
  int32_t styled_for_row(int r) const {
    const int i = rows_env[r];
    if (styles.empty()) return greedy_choice[r];
    const MatchState& m = states[i];
    StyleParams style;
    const size_t base = (size_t(i) * 4 + size_t(m.round.to_move)) * StyleParams::kDim;
    std::memcpy(style.v.data(), styles.data() + base, sizeof(float) * StyleParams::kDim);
    // A per-environment stream derived from the env seed. Mixing the state
    // hash keeps the draw reproducible for a given seed and action sequence
    // without mutating any shared RNG.
    uint64_t style_rng = style_rng_base[i] ^ m.hash();
    return styled_bot(m, cands[i], style, style_rng);
  }

  // A style change while a batch is pending must reach that batch: a match can
  // restart inside pending(), and the opponent source only learns of it after
  // pending() returned. Recomputes in place, so spans and numpy views of
  // styled_choice already handed out see the new values.
  void refresh_styled_choice() {
    if (!batch_ready) return;
    const int rows = static_cast<int>(rows_env.size());
    pool.run([&](int shard, int shards) {
      for (int r = shard; r < rows; r += shards) styled_choice[r] = styled_for_row(r);
    });
  }

  void apply(int i, const Action& action, bool forced) {
    if (!cfg.log_public_actions || (cfg.log_env_limit >= 0 && i >= cfg.log_env_limit)) {
      engine.apply(states[i], action);
      return;
    }
    const MatchState& m = states[i];
    PublicActionEvent event;
    event.env_id = i;
    event.match_id = match_ids[i];
    event.round_index = m.round_index;
    event.step = m.round.steps;
    event.seat = m.round.to_move;
    event.phase = m.round.phase;
    event.action = action;
    event.forced = forced;
    encode_action(action, m.round, event.seat, event.encoded_action);
    // Tribute candidate flags describe the actor's private hand and must not
    // enter a public history stream even when tribute events are requested.
    std::fill(event.encoded_action.begin() + kActTributeFlags,
              event.encoded_action.end(), 0.0f);
    engine.apply(states[i], action);
    event.cards_left = states[i].round.hands[event.seat].size();
    events[i].push_back(event);
  }

  // Advance one environment until it needs a decision. Round ends are drained
  // into the shard's own buffer so that threads never touch shared state.
  void advance(int i, std::vector<RoundResult>& out) {
    MatchState& m = states[i];
    for (int guard = 0; guard < 4096; ++guard) {
      if (m.round.phase == Phase::RoundEnd) {
        RoundResult r;
        engine.end_round(m, r);
        r.env_id = i;
        r.match_id = match_ids[i];
        out.push_back(r);
        if (m.winner >= 0) {
          ++match_ids[i];
          engine.new_match(m, splitmix64(rngs[i]));
        } else {
          engine.begin_round(m);
        }
        continue;
      }
      cands[i].clear();
      engine.legal_actions(m, cands[i]);
      if (cands[i].size() == 1 && cands[i][0].is_pass()) {
        apply(i, cands[i][0], true);
        continue;
      }
      // Do not silently discard a trajectory if an invalid explicit deal or
      // an engine regression produces no actions. pending() reports it below.
      if (cands[i].empty()) { waiting[i] = 0; return; }
      waiting[i] = 1;
      return;
    }
    waiting[i] = 0;
  }
};

VecEnv::VecEnv(int num_envs, int num_threads, EnvConfig cfg, uint64_t seed)
    : impl_(nullptr) {
  if (num_envs <= 0) throw std::invalid_argument("num_envs must be positive");
  if (num_threads <= 0) throw std::invalid_argument("num_threads must be positive");
  if (cfg.log_env_limit < -1) throw std::invalid_argument("log_env_limit must be -1 or nonnegative");
  impl_ = std::make_unique<Impl>(num_envs, num_threads, cfg, seed);
}
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
  s.drained.clear();
  s.rows_env.clear();
  std::fill(s.match_ids.begin(), s.match_ids.end(), 0);
  for (auto& events : s.events) events.clear();
  s.drained_events.clear();
  s.initialized = true;
  s.batch_ready = false;
}

DecisionBatch VecEnv::pending() {
  Impl& s = *impl_;
  if (!s.initialized) throw std::logic_error("reset() must precede pending()");
  const int n = static_cast<int>(s.states.size());
  for (auto& shard : s.finished_shards) shard.clear();

  s.pool.run([&](int shard, int shards) {
    auto& out = s.finished_shards[shard];
    for (int i = shard; i < n; i += shards)
      if (!s.waiting[i]) s.advance(i, out);
  });
  for (auto& shard : s.finished_shards)
    s.finished.insert(s.finished.end(), shard.begin(), shard.end());
  for (int i = 0; i < n; ++i)
    if (!s.waiting[i]) throw std::runtime_error("environment has no legal decision; check explicit deal");

  // Lay the rows out in environment order so that the batch is deterministic
  // however the work was sharded.
  s.rows_env.clear();
  s.offsets.clear();
  s.offsets.push_back(0);
  for (int i = 0; i < n; ++i) {
    if (!s.waiting[i]) continue;
    s.rows_env.push_back(i);
    if (s.cands[i].size() > size_t(std::numeric_limits<int32_t>::max() - s.offsets.back()))
      throw std::overflow_error("candidate batch exceeds int32 offsets");
    s.offsets.push_back(s.offsets.back() + static_cast<int32_t>(s.cands[i].size()));
  }
  const int rows = static_cast<int>(s.rows_env.size());

  s.env_id.resize(rows);
  s.seat.resize(rows);
  s.phase.resize(rows);
  s.round_index.resize(rows);
  s.batch_match_id.resize(rows);
  s.greedy_choice.resize(rows);
  s.styled_choice.resize(rows);
  s.hidden_counts.resize(size_t(rows) * 3 * kNumCardIds);
  if (s.cfg.encode) {
    // No zero fill: the encoders below write every element of their rows.
    s.obs.resize(size_t(rows) * kObsDim);
    s.cand.resize(size_t(s.offsets.back()) * kActDim);
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
      s.round_index[r] = m.round_index;
      s.batch_match_id[r] = s.match_ids[i];
      uint64_t bot_rng = m.rng; // inspecting a batch must never alter shuffle RNG
      s.greedy_choice[r] = greedy_bot(m, s.cands[i], bot_rng);
      s.styled_choice[r] = s.styled_for_row(r);
      for (int rel = 1; rel <= 3; ++rel)
        for (int card = 0; card < kNumCardIds; ++card)
          s.hidden_counts[(size_t(r) * 3 + rel - 1) * kNumCardIds + card] =
              m.round.hands[(m.round.to_move + rel) % 4].count(static_cast<CardId>(card));
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
  b.round_index = s.round_index;
  b.match_id = s.batch_match_id;
  b.greedy_choice = s.greedy_choice;
  b.styled_choice = s.styled_choice;
  b.hidden_counts = s.hidden_counts;
  s.batch_ready = true;
  return b;
}

void VecEnv::step(std::span<const int32_t> choice_index) {
  Impl& s = *impl_;
  const int rows = static_cast<int>(s.rows_env.size());
  if (!s.batch_ready) throw std::logic_error("pending() must precede each step()");
  if (choice_index.size() != size_t(rows))
    throw std::invalid_argument("choices must contain one index per pending row");
  // Validate every choice before any worker mutates state.
  for (int r = 0; r < rows; ++r)
    if (choice_index[r] < 0 || size_t(choice_index[r]) >= s.cands[s.rows_env[r]].size())
      throw std::out_of_range("choice index outside candidate row");
  s.pool.run([&](int shard, int shards) {
    for (int r = shard; r < rows; r += shards) {
      const int i = s.rows_env[r];
      const auto& list = s.cands[i];
      const int k = choice_index[r];
      s.apply(i, list[k], false);
      s.waiting[i] = 0;
    }
  });
  s.batch_ready = false;
}

void VecEnv::set_styles(std::span<const float> styles) {
  Impl& s = *impl_;
  const size_t want = s.states.size() * 4 * StyleParams::kDim;
  if (styles.size() != want)
    throw std::invalid_argument("styles must be [num_envs, 4, STYLE_DIM] float32");
  s.styles.assign(styles.begin(), styles.end());
  s.refresh_styled_choice();
}

void VecEnv::clear_styles() {
  impl_->styles.clear();
  impl_->refresh_styled_choice();
}

std::span<const RoundResult> VecEnv::drain_finished_rounds() {
  Impl& s = *impl_;
  s.drained.swap(s.finished);
  s.finished.clear();
  return s.drained;
}

std::vector<int> VecEnv::fork(int env_id, int copies) {
  Impl& s = *impl_;
  if (!s.initialized) throw std::logic_error("reset() must precede fork()");
  if (env_id < 0 || size_t(env_id) >= s.states.size())
    throw std::out_of_range("env_id outside environment slots");
  if (copies < 0 || size_t(copies) > size_t(std::numeric_limits<int>::max()) - s.states.size())
    throw std::invalid_argument("invalid number of fork copies");
  std::vector<int> ids;
  ids.reserve(copies);
  for (int c = 0; c < copies; ++c) {
    const int id = static_cast<int>(s.states.size());
    s.states.push_back(s.states[env_id]);
    s.rngs.push_back(splitmix64(s.rngs[env_id]));
    s.match_ids.push_back(s.match_ids[env_id]);
    s.cands.push_back(s.cands[env_id]);
    s.waiting.push_back(s.waiting[env_id]);
    s.events.emplace_back();
    // A fresh stream for the copy that leaves the source environment's own
    // stream untouched, so forking never changes the parent's decisions.
    uint64_t derived = s.style_rng_base[env_id] + uint64_t(c) + 1ULL;
    s.style_rng_base.push_back(splitmix64(derived) | 1ULL);
    if (!s.styles.empty()) {
      constexpr size_t kRow = 4 * StyleParams::kDim;
      const std::array<float, kRow> row = [&] {
        std::array<float, kRow> out{};
        std::memcpy(out.data(), s.styles.data() + size_t(env_id) * kRow,
                    sizeof(float) * kRow);
        return out;
      }();
      s.styles.insert(s.styles.end(), row.begin(), row.end());
    }
    ids.push_back(id);
  }
  s.batch_ready = false;
  return ids;
}

std::span<const PublicActionEvent> VecEnv::drain_public_actions() {
  Impl& s = *impl_;
  s.drained_events.clear();
  for (auto& events : s.events) {
    s.drained_events.insert(s.drained_events.end(), events.begin(), events.end());
    events.clear();
  }
  return s.drained_events;
}

int VecEnv::num_envs() const { return static_cast<int>(impl_->states.size()); }
const EnvConfig& VecEnv::config() const { return impl_->cfg; }
const std::vector<Action>& VecEnv::row_actions(int row) const {
  if (!impl_->batch_ready) throw std::logic_error("pending() must precede row_actions()");
  if (row < 0 || size_t(row) >= impl_->rows_env.size())
    throw std::out_of_range("row outside pending batch");
  return impl_->cands[impl_->rows_env[row]];
}

}  // namespace gd
