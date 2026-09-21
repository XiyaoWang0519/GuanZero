// Observation and action encoding, v1. See docs/DESIGN.md section 6.
// Every block below is a fixed slice of the observation vector. The offsets are
// part of the contract: the golden tests in tests/test_encoder_golden.py and
// cpp/tests/test_encoder.cpp pin them.
#pragma once

#include <cstdint>
#include <span>

#include "gd/action.h"
#include "gd/state.h"

namespace gd {

// ---- action encoding ------------------------------------------------------
// [0, 54)     card present at least once
// [54, 108)   card present twice
// [108, 121)  type one-hot, Type::Pass .. Type::BackTribute
// [121, 136)  key one-hot, 0..14 (power, or window index for sequence types)
// [136, 143)  bomb size one-hot, sizes 4..10
// [143, 146)  wild cards used one-hot, 0..2
// [146, 154)  tribute candidate flags, zero outside the tribute phases:
//             copies held == 1, copies held == 2, giving it breaks a pair,
//             breaks a triple, breaks a bomb, the card is SF-relevant,
//             the card is a natural 5, the card is a natural 10
inline constexpr int kActCards1 = 0;
inline constexpr int kActCards2 = 54;
inline constexpr int kActType = 108;
inline constexpr int kActKey = 121;
inline constexpr int kActBombSize = 136;
inline constexpr int kActWilds = 143;
inline constexpr int kActTributeFlags = 146;
inline constexpr int kActDim = 154;

// ---- observation ----------------------------------------------------------
// Seats are encoded relative to the acting seat: rel 1 is the next seat to play
// (LHO), rel 2 is the partner, rel 3 is the previous seat (RHO).
inline constexpr int kObsOwnHand = 0;            // 108
inline constexpr int kObsUnseen = 108;           // 108
inline constexpr int kObsPlayed = 216;           // 4 x 108, LHO, partner, RHO, self
inline constexpr int kObsCardsLeft = 648;        // 3 x 28 one-hot, LHO, partner, RHO
inline constexpr int kObsFinishStatus = 732;     // 4 x 4, active or finish position
inline constexpr int kObsLevels = 748;           // 3 x 13, round, own team, other team
inline constexpr int kObsWildHeld = 787;         // 3, one-hot 0..2 wild cards held
inline constexpr int kObsWildUnseen = 790;       // 3, one-hot 0..2 wild cards unseen
inline constexpr int kObsWildFlags = 793;        // 12, see below
inline constexpr int kObsTrick = 805;            // 154 + 4 + 4 + 1 = 163
inline constexpr int kObsTrickTop = 805;         //   154, top play as an action
inline constexpr int kObsTrickHolder = 959;      //   4, one-hot rel seat, index 0 = none
inline constexpr int kObsTrickPasses = 963;      //   4, one-hot passes since the top play
inline constexpr int kObsTrickLeading = 967;     //   1
inline constexpr int kObsLastAction = 968;       // 3 x 154, LHO, partner, RHO
inline constexpr int kObsPhase = 1430;           // 3, play, tribute, back-tribute
inline constexpr int kObsRoles = 1433;           // 6, see below
inline constexpr int kObsTribute = 1439;         // 4 x (54 + 8)
inline constexpr int kObsKnownHoldings = 1687;   // 3 x 54
inline constexpr int kObsDim = 1849;

// Wild flags, 12 bits at kObsWildFlags. The first nine say that a wild card in
// hand can complete a play of that type; the last three are hand properties.
//  0 pair, 1 triple, 2 full house, 3 straight, 4 tube, 5 plate,
//  6 bomb of four, 7 bomb of five or more, 8 straight flush,
//  9 the hand holds a natural level card, 10 the hand holds both wild cards,
// 11 some bomb the hand can play needs a wild card.

// Roles, 6 bits at kObsRoles: own finishing position last round one-hot over
// {Banker, Follower, Third, Dweller, none} (5), then whether this round's
// tribute counterpart is the partner (1).

// Tribute context, 4 slots of 62 at kObsTribute, one per public exchange of
// this round in the order they happened: the card (54) then payer and receiver
// as relative seats (4 + 4). All zero when no tribute took place.

// Encodes one candidate action. `dst` must have kActDim entries.
void encode_action(const Action& a, const RoundState& r, int seat,
                   std::span<float> dst);

// Encodes the observation of `seat` in `r`. `dst` must have kObsDim entries.
// `m` supplies the team levels, the previous finishing order and the phase.
void encode_observation(const MatchState& m, int seat, std::span<float> dst);

}  // namespace gd
