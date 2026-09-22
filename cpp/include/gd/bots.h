// In-engine heuristic players. Used to bootstrap self-play out of random play
// (DESIGN.md 8.2 item 4), as league opponents and as the fuzzer's driver.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "gd/action.h"
#include "gd/state.h"

namespace gd {

// Uniform over the candidate list.
int random_bot(const MatchState& m, const std::vector<Action>& cands, uint64_t& rng);

// A greedy player. Leads its cheapest play, follows with the cheapest play that
// beats the top, never spends a bomb unless it is the only option or an
// opponent is about to finish, and never overtakes its own partner.
int greedy_bot(const MatchState& m, const std::vector<Action>& cands, uint64_t& rng);

// Tribute and back-tribute heuristics of DESIGN.md 8.3 item 1.
int tribute_bot(const MatchState& m, const std::vector<Action>& cands, uint64_t& rng);

// ---- styled heuristic player (DESIGN.md 7.3, M2_TODO task 1) --------------
// One bot driven by a continuous style vector, so that league and probe
// opponents have habits. Styles are continuous parameters sampled per match,
// never a fixed list of bots. Nothing here changes the rules engine: the style
// only reorders the candidates the engine already generated.
//
// Slot layout of `v`, all documented, none hardcoded elsewhere:
//
//   index                       name              range      neutral
//   0                           bomb_threshold    [0, 1]     1
//   1 .. 1 + kNumPlayTypes - 1  type_pref[Type]   any        0
//   14                          follow_aggression [0, 1]     0
//   15                          lead_high_bias    [-1, 1]    0
//   16                          partner_weight    [0, 1]     1
//   17                          temperature       >= 0       0
//
// The type-preference block has one slot per member of `Type`, in enum order,
// so `type_pref(t)` is simply `v[kTypePref + int(t)]`. The slots of Pass,
// Tribute and BackTribute exist to keep that indexing mechanical; the bot never
// reads them, because a pass is not scored and tribute phases delegate to
// tribute_bot. Preferences are log-weights: the bot's internal score is divided
// by kScoreUnit to form logits, and one unit of type preference is exactly one
// logit, so at temperature 1 a preference of 1 multiplies that play's
// probability by e. That is the multiplicative reading of the weights.
//
// Meaning of each dimension:
//   bomb_threshold    0 spends bombs freely and early, 1 only when the play
//                     empties the hand or an opponent is about to finish
//                     (the greedy rule). It sets the opponent-pressure level
//                     below which the bomb class is unlocked.
//   type_pref         additive log-weight per play type when leading.
//   follow_aggression 0 follows with the cheapest beating play (greedy), 1
//                     prefers high and overwhelming plays when following.
//   lead_high_bias    negative leads low cards first, positive leads high.
//   partner_weight    1 never overtakes the partner (greedy), 0 ignores it.
//                     Between the two it overtakes only while the partner still
//                     holds at least `partner_weight * (kHandSize + 1)` cards.
//   temperature       0 takes the argmax of the internal scores, above 0 draws
//                     from the softmax of score / (kScoreUnit * temperature).
//                     Sampling is deterministic given `rng`.
//
// Trivially copyable, fixed size, no heap, so a whole league of styles is a
// plain float array.
struct StyleParams {
  static constexpr int kNumPlayTypes = static_cast<int>(Type::kNumTypes);  // 13
  static constexpr int kBombThreshold = 0;
  static constexpr int kTypePref = 1;
  static constexpr int kFollowAggression = kTypePref + kNumPlayTypes;      // 14
  static constexpr int kLeadHighBias = kFollowAggression + 1;              // 15
  static constexpr int kPartnerWeight = kLeadHighBias + 1;                 // 16
  static constexpr int kTemperature = kPartnerWeight + 1;                  // 17
  static constexpr int kDim = kTemperature + 1;                            // 18

  // One internal score unit, and therefore one logit. See the note above.
  static constexpr double kScoreUnit = 25000.0;

  std::array<float, kDim> v{};

  float bomb_threshold() const { return v[kBombThreshold]; }
  float type_pref(Type t) const { return v[kTypePref + static_cast<int>(t)]; }
  float follow_aggression() const { return v[kFollowAggression]; }
  float lead_high_bias() const { return v[kLeadHighBias]; }
  float partner_weight() const { return v[kPartnerWeight]; }
  float temperature() const { return v[kTemperature]; }

  // Reproduces greedy_bot exactly: hoards the bomb class, never overtakes the
  // partner, no type or rank bias, argmax rather than sampling.
  static StyleParams neutral() {
    StyleParams s;
    s.v.fill(0.0f);
    s.v[kBombThreshold] = 1.0f;
    s.v[kPartnerWeight] = 1.0f;
    return s;
  }
};
static_assert(StyleParams::kDim == 18, "the Python side pins STYLE_DIM");

// Style-parameterised heuristic. Returns an index into `cands`, always in
// range. Tribute and back-tribute phases delegate to tribute_bot; styles do not
// affect tribute.
int styled_bot(const MatchState& m, const std::vector<Action>& cands,
               const StyleParams& style, uint64_t& rng);

}  // namespace gd
