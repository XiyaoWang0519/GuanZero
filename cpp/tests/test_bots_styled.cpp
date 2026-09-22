// The styled heuristic bot of M2_TODO task 1. The neutral style must reproduce
// greedy_bot exactly, every choice must index a real candidate, and sampling
// must be a function of the seed alone.
#include <cstdio>
#include <vector>

#include "gd/bots.h"
#include "gd/movegen.h"
#include "gd/state.h"
#include "test_util.h"

namespace {

uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Walks random rounds and calls `visit` at every decision point.
template <typename F>
void walk_decisions(long long want, uint64_t seed, F&& visit) {
  gd::Engine engine(gd::RuleConfig::house(), gd::ActionConfig{});
  gd::MatchState m;
  gd::RoundResult res;
  uint64_t rng = seed;
  engine.new_match(m, splitmix64(rng));
  std::vector<gd::Action> cands;
  cands.reserve(4096);
  long long seen = 0;
  while (seen < want) {
    if (m.round.phase == gd::Phase::RoundEnd) {
      engine.end_round(m, res);
      if (m.winner >= 0) engine.new_match(m, splitmix64(rng));
      else engine.begin_round(m);
      continue;
    }
    cands.clear();
    engine.legal_actions(m, cands);
    if (cands.empty()) { engine.new_match(m, splitmix64(rng)); continue; }
    ++seen;
    visit(m, cands);
    // Advance with the greedy line so that the walk visits the states a
    // heuristic bot actually reaches, not only random ones.
    uint64_t bot = m.rng;
    engine.apply(m, cands[gd::greedy_bot(m, cands, bot)]);
  }
}

}  // namespace

TEST("styled bot: the neutral style reproduces greedy_bot") {
  const gd::StyleParams neutral = gd::StyleParams::neutral();
  long long points = 0, mismatches = 0, out_of_range = 0;
  walk_decisions(12000, 0xA11CEULL, [&](const gd::MatchState& m,
                                        const std::vector<gd::Action>& cands) {
    uint64_t r1 = 12345, r2 = 999;
    const int g = gd::greedy_bot(m, cands, r1);
    const int s = gd::styled_bot(m, cands, neutral, r2);
    ++points;
    if (s < 0 || size_t(s) >= cands.size()) ++out_of_range;
    if (g != s) ++mismatches;
  });
  std::fprintf(stderr, "  neutral vs greedy: %lld decision points, %lld mismatches\n",
               points, mismatches);
  CHECK(points >= 10000);
  CHECK_EQ(mismatches, 0LL);
  CHECK_EQ(out_of_range, 0LL);
}

TEST("styled bot: every style keeps the choice inside the candidate list") {
  uint64_t style_rng = 7;
  std::vector<gd::StyleParams> styles;
  for (int k = 0; k < 24; ++k) {
    gd::StyleParams p;
    for (int d = 0; d < gd::StyleParams::kDim; ++d) {
      const double u = double(splitmix64(style_rng) >> 11) / 9007199254740992.0;
      p.v[d] = static_cast<float>(u * 2.0 - 1.0);
    }
    p.v[gd::StyleParams::kTemperature] = static_cast<float>(k % 4) * 0.5f;
    styles.push_back(p);
  }
  long long points = 0, bad = 0;
  walk_decisions(4000, 0xBEEFULL, [&](const gd::MatchState& m,
                                      const std::vector<gd::Action>& cands) {
    for (const auto& p : styles) {
      uint64_t r = 5 + points;
      const int c = gd::styled_bot(m, cands, p, r);
      if (c < 0 || size_t(c) >= cands.size()) ++bad;
    }
    ++points;
  });
  std::fprintf(stderr, "  style sweep: %lld points x %zu styles, %lld out of range\n",
               points, styles.size(), bad);
  CHECK_EQ(bad, 0LL);
}

TEST("styled bot: sampling is deterministic given the seed") {
  gd::StyleParams hot = gd::StyleParams::neutral();
  hot.v[gd::StyleParams::kTemperature] = 0.75f;
  hot.v[gd::StyleParams::kBombThreshold] = 0.4f;
  long long differed = 0, mismatched = 0, points = 0;
  walk_decisions(3000, 0xD1CEULL, [&](const gd::MatchState& m,
                                      const std::vector<gd::Action>& cands) {
    uint64_t a = 424242, b = 424242, c = 777;
    const int x = gd::styled_bot(m, cands, hot, a);
    const int y = gd::styled_bot(m, cands, hot, b);
    const int z = gd::styled_bot(m, cands, hot, c);
    if (x != y) ++mismatched;
    if (x != z) ++differed;
    ++points;
  });
  std::fprintf(stderr, "  sampling: %lld points, %lld seed mismatches, "
                       "%lld differing across seeds\n", points, mismatched, differed);
  CHECK_EQ(mismatched, 0LL);
  CHECK(differed > 0);          // the temperature really does sample
}

TEST("styled bot: extreme styles move the choice away from greedy") {
  gd::StyleParams bomber = gd::StyleParams::neutral();
  bomber.v[gd::StyleParams::kBombThreshold] = 0.0f;
  gd::StyleParams high = gd::StyleParams::neutral();
  high.v[gd::StyleParams::kLeadHighBias] = 1.0f;
  high.v[gd::StyleParams::kFollowAggression] = 1.0f;
  gd::StyleParams loner = gd::StyleParams::neutral();
  loner.v[gd::StyleParams::kPartnerWeight] = 0.0f;
  long long bomb_diff = 0, high_diff = 0, lone_diff = 0;
  walk_decisions(6000, 0xF00DULL, [&](const gd::MatchState& m,
                                      const std::vector<gd::Action>& cands) {
    uint64_t r0 = 1, r1 = 1, r2 = 1, r3 = 1;
    const int g = gd::styled_bot(m, cands, gd::StyleParams::neutral(), r0);
    if (gd::styled_bot(m, cands, bomber, r1) != g) ++bomb_diff;
    if (gd::styled_bot(m, cands, high, r2) != g) ++high_diff;
    if (gd::styled_bot(m, cands, loner, r3) != g) ++lone_diff;
  });
  std::fprintf(stderr, "  differences from neutral: bomb %lld, high %lld, partner %lld\n",
               bomb_diff, high_diff, lone_diff);
  CHECK(bomb_diff > 0);
  CHECK(high_diff > 0);
  CHECK(lone_diff > 0);
}

TEST("styled bot: the style layout is the documented one") {
  CHECK_EQ(int(gd::StyleParams::kDim), 18);
  CHECK_EQ(int(gd::StyleParams::kBombThreshold), 0);
  CHECK_EQ(int(gd::StyleParams::kTypePref), 1);
  CHECK_EQ(int(gd::StyleParams::kFollowAggression), 14);
  CHECK_EQ(int(gd::StyleParams::kLeadHighBias), 15);
  CHECK_EQ(int(gd::StyleParams::kPartnerWeight), 16);
  CHECK_EQ(int(gd::StyleParams::kTemperature), 17);
  const gd::StyleParams n = gd::StyleParams::neutral();
  CHECK_EQ(n.bomb_threshold(), 1.0f);
  CHECK_EQ(n.partner_weight(), 1.0f);
  CHECK_EQ(n.follow_aggression(), 0.0f);
  CHECK_EQ(n.lead_high_bias(), 0.0f);
  CHECK_EQ(n.temperature(), 0.0f);
  CHECK_EQ(n.type_pref(gd::Type::Straight), 0.0f);
  CHECK(std::is_trivially_copyable<gd::StyleParams>::value);
}
