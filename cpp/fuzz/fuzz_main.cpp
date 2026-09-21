// M0 task 7: the property fuzzer. Plays random rounds and checks every
// invariant of docs/RULES.md section 14 after every step.
//
//   ./build/cpp/fuzz/gd_fuzz --rounds 10000000 --threads 14
//   ./build-asan/cpp/fuzz/gd_fuzz --rounds 100000 --threads 1
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "gd/bots.h"
#include "gd/movegen.h"
#include "gd/rules.h"
#include "gd/state.h"

namespace {

std::atomic<long long> g_failures{0};
std::atomic<long long> g_rounds{0};
std::atomic<long long> g_decisions{0};

void report(const char* what, const gd::MatchState& m) {
  if (g_failures.fetch_add(1) < 20) {
    std::fprintf(stderr, "INVARIANT %s  phase=%d seat=%d level=%d finished=%d\n",
                 what, static_cast<int>(m.round.phase), m.round.to_move,
                 m.round.level, m.round.num_finished);
  }
}

uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Invariant 1: every card id is present exactly twice across hands and plays.
void check_conservation(const gd::MatchState& m) {
  int counts[gd::kNumCardIds] = {0};
  for (int s = 0; s < 4; ++s) {
    for (int c = 0; c < gd::kNumCardIds; ++c) {
      counts[c] += m.round.hands[s].count(static_cast<gd::CardId>(c));
      counts[c] += m.round.played[s].count(static_cast<gd::CardId>(c));
    }
  }
  for (int c = 0; c < gd::kNumCardIds; ++c)
    if (counts[c] != 2) { report("conservation", m); return; }
}

// Invariants 2 and 3: the legal set has the right shape and every member is a
// real reading that beats the top play.
void check_legal_set(const gd::MatchState& m, const std::vector<gd::Action>& cands,
                     const gd::RuleConfig& rules, bool deep) {
  if (cands.empty()) { report("empty legal set", m); return; }
  if (m.round.phase != gd::Phase::Play) return;
  const bool leading = m.round.top.is_pass();
  int passes = 0;
  for (const auto& a : cands) passes += a.is_pass() ? 1 : 0;
  if (leading && passes != 0) report("pass offered on a lead", m);
  if (!leading && passes != 1) report("pass missing when following", m);

  const gd::Hand& hand = m.round.hands[m.round.to_move];
  for (const auto& a : cands) {
    if (a.is_pass()) continue;
    if (!hand.contains(a.cards)) { report("action uses cards not held", m); continue; }
    if (!gd::beats(a, m.round.top)) { report("action does not beat the top", m); continue; }
    if (!deep) continue;              // re-reading every candidate is the slow part
    bool found = false;
    for (const auto& r : gd::interpret(a.cards, m.round.level, rules))
      if (r.type == a.type && r.key == a.key && r.bomb_size == a.bomb_size) found = true;
    if (!found) report("action is not a legal reading of its cards", m);
  }
}

// Invariant 7: the finishing order never repeats a seat and a finished seat
// never moves again.
void check_order(const gd::MatchState& m) {
  bool seen[4] = {false, false, false, false};
  for (int i = 0; i < m.round.num_finished; ++i) {
    const int s = m.round.order[i];
    if (s < 0 || s > 3 || seen[s]) { report("bad finishing order", m); return; }
    seen[s] = true;
    // Only the seats that actually ran out are empty; the tail is ranked at
    // round end and keeps its cards (RULES.md 7.3).
    if (i < m.round.num_out && !m.round.hands[s].empty())
      report("finished seat still holds cards", m);
  }
  if (m.round.phase == gd::Phase::Play && !m.round.active(m.round.to_move))
    report("a finished seat is to move", m);
}

void worker(long long rounds, uint64_t seed, bool canonical, bool verbose, int deep_every) {
  gd::RuleConfig rules = gd::RuleConfig::house();
  gd::ActionConfig acfg = canonical ? gd::ActionConfig{} : gd::ActionConfig::full();
  gd::Engine engine(rules, acfg);
  std::vector<gd::Action> cands;
  cands.reserve(8192);
  gd::MatchState m;
  gd::RoundResult res;
  uint64_t rng = seed;
  engine.new_match(m, splitmix64(rng));
  std::array<int8_t, 2> prev_levels = m.levels;
  long long done = 0;
  int round_steps = 0;

  while (done < rounds) {
    if (m.round.phase == gd::Phase::RoundEnd) {
      engine.end_round(m, res);
      // Invariant 10: levels never decrease except through the A fail reset.
      for (int t = 0; t < 2; ++t)
        if (m.levels[t] < prev_levels[t] && m.fails[t] != 0)
          report("level decreased outside a reset", m);
      prev_levels = m.levels;
      ++done;
      g_rounds.fetch_add(1, std::memory_order_relaxed);
      round_steps = 0;
      if (m.winner >= 0) {
        engine.new_match(m, splitmix64(rng));
        prev_levels = m.levels;
      } else {
        engine.begin_round(m);
      }
      check_conservation(m);
      continue;
    }

    cands.clear();
    engine.legal_actions(m, cands);
    const bool deep = deep_every > 0 && (splitmix64(rng) % uint64_t(deep_every)) == 0;
    check_legal_set(m, cands, rules, deep);
    if (cands.empty()) { engine.new_match(m, splitmix64(rng)); continue; }
    g_decisions.fetch_add(1, std::memory_order_relaxed);

    // Invariant 8: the same state and action give the same hash everywhere.
    gd::MatchState copy;
    if (deep) {
      const uint64_t before = m.hash();
      copy = m;
      if (copy.hash() != before) report("copy changed the hash", m);
    }

    const gd::Action& pick = cands[splitmix64(rng) % cands.size()];
    engine.apply(m, pick);

    if (deep) {
      gd::MatchState replay = copy;
      engine.apply(replay, pick);
      if (replay.hash() != m.hash()) report("apply is not deterministic", m);
    }

    check_conservation(m);
    check_order(m);

    // Invariant 6: a round of random play ends in fewer than 600 steps.
    if (m.round.phase == gd::Phase::Play && ++round_steps >= 600)
      { report("round did not terminate", m); engine.new_match(m, splitmix64(rng)); round_steps = 0; }
  }
  if (verbose) std::fprintf(stderr, "worker %llu done\n", (unsigned long long)seed);
}

}  // namespace

int main(int argc, char** argv) {
  long long rounds = 100000;
  int threads = 1;
  bool canonical = true;
  bool verbose = false;
  int deep_every = 64;      // run the costly re-reading checks on 1 decision in N
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--full")) canonical = false;
    else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
    else if (!std::strcmp(argv[i], "--deep-every") && i + 1 < argc) deep_every = std::atoi(argv[++i]);
  }
  if (threads < 1) threads = 1;
  const long long per = (rounds + threads - 1) / threads;

  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back(worker, per, 0x5EED0000ULL + uint64_t(t) * 7919ULL, canonical,
                      verbose, deep_every);
  for (auto& th : pool) th.join();

  std::printf("rounds=%lld decisions=%lld failures=%lld mode=%s deep_every=%d\n",
              g_rounds.load(), g_decisions.load(), g_failures.load(),
              canonical ? "canonical" : "full", deep_every);
  return g_failures.load() == 0 ? 0 : 1;
}
