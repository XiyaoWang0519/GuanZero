// M0 task 9: throughput of the rules engine under random play.
// Target (DESIGN.md 3): at least 100,000 decisions per second per core in
// canonical mode. Also records the candidate-count statistics that task 5 asks
// for, in both enumeration modes.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "gd/movegen.h"
#include "gd/state.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Stats {
  long long decisions = 0;
  long long candidates = 0;
  long long rounds = 0;
  long long matches = 0;
  std::vector<int> widths;   // candidates per decision, sampled
};

uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

Stats run_random(const gd::Engine& engine, uint64_t seed, double seconds,
                 bool collect_widths) {
  Stats st;
  std::vector<gd::Action> cands;
  cands.reserve(4096);
  gd::MatchState m;
  engine.new_match(m, seed);
  gd::RoundResult res;
  uint64_t rng = seed * 2654435761ULL + 1;
  const auto deadline = Clock::now() + std::chrono::duration<double>(seconds);
  int check = 0;
  while (true) {
    if ((++check & 0xFF) == 0 && Clock::now() > deadline) break;
    if (m.round.phase == gd::Phase::RoundEnd) {
      engine.end_round(m, res);
      ++st.rounds;
      if (m.winner >= 0) {
        ++st.matches;
        engine.new_match(m, splitmix64(rng));
      } else {
        engine.begin_round(m);
      }
      continue;
    }
    cands.clear();
    engine.legal_actions(m, cands);
    if (cands.empty()) {
      std::fprintf(stderr, "no legal actions, phase %d seat %d\n",
                   static_cast<int>(m.round.phase), m.round.to_move);
      break;
    }
    ++st.decisions;
    st.candidates += static_cast<long long>(cands.size());
    if (collect_widths && (st.decisions % 16) == 0)
      st.widths.push_back(static_cast<int>(cands.size()));
    engine.apply(m, cands[splitmix64(rng) % cands.size()]);
  }
  return st;
}

void report(const char* label, const Stats& st, double seconds, int threads) {
  const double dps = st.decisions / seconds;
  std::printf("%-28s %10.0f dec/s  %8.0f dec/s/core  %9lld rounds  %7lld matches"
              "  %6.1f cand/dec\n",
              label, dps, dps / threads, st.rounds, st.matches,
              st.decisions ? double(st.candidates) / double(st.decisions) : 0.0);
}

void percentiles(const char* label, std::vector<int> w) {
  if (w.empty()) return;
  std::sort(w.begin(), w.end());
  auto q = [&](double p) { return w[std::min(w.size() - 1, size_t(p * w.size()))]; };
  double mean = 0;
  for (int x : w) mean += x;
  mean /= double(w.size());
  std::printf("%-28s mean %6.1f  p50 %5d  p90 %5d  p99 %5d  max %5d  (n=%zu)\n",
              label, mean, q(0.50), q(0.90), q(0.99), w.back(), w.size());
}

}  // namespace

int main(int argc, char** argv) {
  double seconds = 3.0;
  int threads = 1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
  }

  gd::RuleConfig rules = gd::RuleConfig::house();
  gd::ActionConfig canonical;
  gd::ActionConfig full = gd::ActionConfig::full();

  std::printf("# single core, %.1fs per configuration\n", seconds);
  {
    gd::Engine e(rules, canonical);
    Stats st = run_random(e, 1, seconds, true);
    report("canonical, 1 core", st, seconds, 1);
    percentiles("  candidates, canonical", st.widths);
  }
  {
    gd::Engine e(rules, full);
    Stats st = run_random(e, 2, seconds, true);
    report("full, 1 core", st, seconds, 1);
    percentiles("  candidates, full", st.widths);
  }

  if (threads > 1) {
    std::printf("# %d threads\n", threads);
    std::vector<std::thread> pool;
    std::vector<Stats> out(threads);
    const auto t0 = Clock::now();
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        gd::Engine e(rules, canonical);
        out[t] = run_random(e, 100 + t, seconds, false);
      });
    }
    for (auto& th : pool) th.join();
    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
    Stats total;
    for (auto& s : out) {
      total.decisions += s.decisions;
      total.candidates += s.candidates;
      total.rounds += s.rounds;
      total.matches += s.matches;
    }
    report("canonical, all cores", total, wall, threads);
  }
  return 0;
}
