// Combination types, concrete actions and the 393 abstract action ids.
// See docs/RULES.md sections 5, 6 and 11.
#pragma once

#include <cstdint>
#include <string>

#include "gd/cards.h"

namespace gd {

enum class Type : uint8_t {
  Pass = 0,
  Single = 1,
  Pair = 2,
  Triple = 3,
  FullHouse = 4,
  Straight = 5,
  Tube = 6,
  Plate = 7,
  Bomb = 8,
  StraightFlush = 9,
  JokerBomb = 10,
  Tribute = 11,       // phase actions, never legal during play
  BackTribute = 12,
  kNumTypes = 13,
};

const char* type_name(Type t);
constexpr bool is_bomb_class(Type t) {
  return t == Type::Bomb || t == Type::StraightFlush || t == Type::JokerBomb;
}

// A concrete action: a reading plus the exact cards.
// Trivially copyable. `key` is the power for Single, Pair, Triple, FullHouse and
// Bomb, and the window index for Straight, Tube, Plate and StraightFlush.
// `bomb_size` is the card count of a Bomb and 0 otherwise.
// For Tribute and BackTribute the single card sits in `cards` and key is its power.
struct Action {
  Type type = Type::Pass;
  int8_t key = 0;
  int8_t bomb_size = 0;
  int8_t wilds = 0;        // wild cards consumed by this reading
  // FullHouse only: the pair's rank in the POWER domain, matching `key`
  // (0..12, 13 = BJ pair, 14 = RJ pair). Power, not natural rank, so that
  // abstract_id() needs no round level.
  int8_t fh_pair_rank = -1;
  Hand cards{};

  constexpr bool is_pass() const { return type == Type::Pass; }
  constexpr bool operator==(const Action& o) const = default;
  std::string to_string() const;
};

// Total order inside the bomb class (RULES.md 6.4). Larger wins, equal never wins.
// 4-bomb < 5-bomb < straight flush < 6-bomb < ... < 10-bomb < joker bomb.
struct Strength {
  int cls = 0;
  int key = 0;
  constexpr bool operator<(const Strength& o) const {
    return cls != o.cls ? cls < o.cls : key < o.key;
  }
};
constexpr Strength strength(const Action& a) {
  if (a.type == Type::JokerBomb) return {999, 0};
  if (a.type == Type::StraightFlush) return {11, a.key};   // between 5-bomb and 6-bomb
  return {a.bomb_size * 2, a.key};                          // 4-bomb = 8, 10-bomb = 20
}

// True if `cand` may be played over `top`. A pass-typed `top` means leading.
bool beats(const Action& cand, const Action& top);
bool beats_reading(Type ct, int ckey, int cbomb, Type tt, int tkey, int tbomb);

// ---- abstract actions (RULES.md 11.1) ------------------------------------
// 393 ids laid out in this order:
//   0                pass
//   1   .. 15        single, power 0..14
//   16  .. 30        pair, power 0..14
//   31  .. 43        triple, power 0..12
//   44  .. 225       full house, 13 triple powers x 14 pair slots
//   226 .. 235       straight, window 0..9
//   236 .. 247       tube, window 0..11
//   248 .. 260       plate, window 0..12
//   261 .. 351       bomb, 13 powers x sizes 4..10  (power * 7 + (size - 4))
//   352 .. 391       straight flush, window * 4 + suit
//   392              joker bomb
// The full house pair slot encodes the pair's power: powers 0..12 other than
// the triple's power keep their order (0..11), BJ pair is 12 and RJ pair is 13.
inline constexpr int kNumAbstract = 393;
inline constexpr int kAbstractPass = 0;
inline constexpr int kAbstractSingle = 1;
inline constexpr int kAbstractPair = 16;
inline constexpr int kAbstractTriple = 31;
inline constexpr int kAbstractFullHouse = 44;
inline constexpr int kAbstractStraight = 226;
inline constexpr int kAbstractTube = 236;
inline constexpr int kAbstractPlate = 248;
inline constexpr int kAbstractBomb = 261;
inline constexpr int kAbstractStraightFlush = 352;
inline constexpr int kAbstractJokerBomb = 392;

int abstract_id(const Action& a);
// Inverse without cards: fills type, key, bomb_size and fh_pair_rank.
Action abstract_action(int id);

}  // namespace gd
