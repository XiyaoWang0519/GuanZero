// RULES.md section 12 test vectors that do not need the round/match state
// machine, plus abstract action id round trips.
#include <cstring>
#include <set>
#include <tuple>
#include <vector>

#include "gd/action.h"
#include "gd/cards.h"
#include "gd/rules.h"
#include "test_util.h"

using namespace gd;

namespace {

constexpr const char* kRanks = "23456789TJQKA";
int rank_index(char c) { return static_cast<int>(std::strchr(kRanks, c) - kRanks); }

using RSet = std::set<std::tuple<Type, int, int>>;

RSet to_set(const std::vector<Reading>& rs) {
  RSet s;
  for (const Reading& r : rs) s.insert({r.type, r.key, r.bomb_size});
  return s;
}

RSet interp(const std::string& cards, int level) {
  return to_set(interpret(Hand::from_string(cards), level));
}

RSet best(const std::string& cards, int level) {
  return to_set(best_readings(Hand::from_string(cards), level));
}

Reading only(const std::string& cards, int level) {
  const auto rs = interpret(Hand::from_string(cards), level);
  return rs.empty() ? Reading{} : rs.front();
}

bool has(const RSet& s, Type t, int key, int bomb = 0) { return s.count({t, key, bomb}) != 0; }

}  // namespace

// ---- Ordering and singles --------------------------------------------------

TEST("T-ORD-01") {
  const int L = rank_index('7');
  const Reading s7 = only("S7", L);
  const Reading sa = only("SA", L);
  CHECK(beats_reading(s7.type, s7.key, s7.bomb_size, sa.type, sa.key, sa.bomb_size));
}

TEST("T-ORD-02") {
  const int L = rank_index('7');
  const Reading sb = only("SB", L);
  const Reading s7 = only("S7", L);
  const Reading hr = only("HR", L);
  CHECK_EQ((int)sb.key, 13);
  CHECK_EQ((int)hr.key, 14);
  CHECK(beats_reading(sb.type, sb.key, sb.bomb_size, s7.type, s7.key, s7.bomb_size));
  CHECK(beats_reading(hr.type, hr.key, hr.bomb_size, sb.type, sb.key, sb.bomb_size));
}

TEST("T-ORD-03") {
  const int L = rank_index('7');
  const Reading h7 = only("H7", L);
  const Reading s7 = only("S7", L);
  CHECK_EQ((int)h7.key, 12);
  CHECK_EQ((int)h7.key, (int)s7.key);
  CHECK(!beats_reading(h7.type, h7.key, h7.bomb_size, s7.type, s7.key, s7.bomb_size));
  CHECK(!beats_reading(s7.type, s7.key, s7.bomb_size, h7.type, h7.key, h7.bomb_size));
}

TEST("T-ORD-04") {
  const int L = rank_index('7');
  CHECK_EQ((int)only("S8", L).key, 5);
  CHECK_EQ((int)only("S6", L).key, 4);
}

TEST("T-ORD-05") {
  const int L = rank_index('A');
  CHECK_EQ((int)only("SA", L).key, 12);
  CHECK_EQ((int)only("SK", L).key, 11);
}

// ---- Pairs and triples ------------------------------------------------------

TEST("T-PAIR-01") {
  const int L = rank_index('7');
  const auto r = interpret(Hand::from_string("H7 SA"), L);
  CHECK_EQ(r.size(), (size_t)1);
  CHECK(r[0].type == Type::Pair && r[0].key == 11);
}

TEST("T-PAIR-02") {
  CHECK(interpret(Hand::from_string("H7 SB"), rank_index('7')).empty());
}

TEST("T-PAIR-03") {
  CHECK(interpret(Hand::from_string("SB HR"), rank_index('7')).empty());
}

TEST("T-PAIR-04") {
  const int L = rank_index('7');
  const auto a = interpret(Hand::from_string("SB SB"), L);
  CHECK(a.size() == 1 && a[0].type == Type::Pair && a[0].key == 13);
  const auto b = interpret(Hand::from_string("HR HR"), L);
  CHECK(b.size() == 1 && b[0].type == Type::Pair && b[0].key == 14);
}

TEST("T-PAIR-05") {
  const int L = rank_index('7');
  const auto r = interpret(Hand::from_string("H7 H7"), L);
  CHECK(r.size() == 1 && r[0].type == Type::Pair && r[0].key == 12);
}

TEST("T-PAIR-06") {
  const int L = rank_index('7');
  const auto r = interpret(Hand::from_string("H7 S7"), L);
  CHECK(r.size() == 1 && r[0].type == Type::Pair && r[0].key == 12);
}

TEST("T-TRI-01") {
  const int L = rank_index('7');
  const auto r = interpret(Hand::from_string("H7 H7 S9"), L);
  CHECK(r.size() == 1 && r[0].type == Type::Triple && r[0].key == 6);
}

TEST("T-TRI-02") {
  CHECK(interpret(Hand::from_string("SB SB H7"), rank_index('7')).empty());
}

// ---- Full houses -------------------------------------------------------------

TEST("T-FH-01") {
  const int L = rank_index('7');
  const RSet r = interp("S5 D5 C5 SB SB", L);
  CHECK(has(r, Type::FullHouse, 3));
  CHECK_EQ(r.size(), (size_t)1);
}

TEST("T-FH-02") {
  CHECK(interpret(Hand::from_string("S5 D5 C5 SB H7"), rank_index('7')).empty());
}

TEST("T-FH-03") {
  const int L = rank_index('7');
  const RSet r = interp("S5 D5 H7 S9 D9", L);
  CHECK(has(r, Type::FullHouse, 3));
  CHECK(has(r, Type::FullHouse, 6));
  const RSet b = best("S5 D5 H7 S9 D9", L);
  CHECK(has(b, Type::FullHouse, 6));
  CHECK(!has(b, Type::FullHouse, 3));
}

TEST("T-FH-04") {
  const int L = rank_index('7');
  const RSet r = interp("H7 H7 S9 D9 C9", L);
  CHECK(has(r, Type::Bomb, 6, 5));
  CHECK(has(r, Type::FullHouse, 6));
  const RSet b = best("H7 H7 S9 D9 C9", L);
  CHECK(has(b, Type::Bomb, 6, 5));
  CHECK(has(b, Type::FullHouse, 6));
}

TEST("T-FH-05") {
  const int L = rank_index('7');
  const Reading a = interpret(Hand::from_string("S7 D7 C7 S2 D2"), L).front();
  const Reading b = interpret(Hand::from_string("SA DA CA SK DK"), L).front();
  CHECK_EQ((int)a.key, 12);
  CHECK_EQ((int)b.key, 11);
  CHECK(beats_reading(a.type, a.key, a.bomb_size, b.type, b.key, b.bomb_size));
}

// ---- Straights and straight flushes ------------------------------------------

TEST("T-STR-01") {
  const int L = rank_index('7');
  const RSet r = interp("SA D2 C3 S4 H5", L);
  CHECK(has(r, Type::Straight, 0));
  CHECK_EQ(r.size(), (size_t)1);
}

TEST("T-STR-02") {
  CHECK(interpret(Hand::from_string("SK DA C2 S3 H4"), rank_index('7')).empty());
}

TEST("T-STR-03") {
  const int L = rank_index('7');
  const RSet r = interp("ST DJ CQ SK HA", L);
  CHECK(has(r, Type::Straight, 9));
  CHECK_EQ(r.size(), (size_t)1);
}

TEST("T-STR-04") {
  const int L = rank_index('7');
  const RSet r = interp("S5 D6 S7 C8 D9", L);
  CHECK(has(r, Type::Straight, 4));
  CHECK_EQ(r.size(), (size_t)1);
}

TEST("T-STR-05") {
  const int L = rank_index('7');
  const RSet r = interp("SA S2 S3 S4 S5", L);
  CHECK(has(r, Type::StraightFlush, 0));
  CHECK_EQ(r.size(), (size_t)1);  // only the flush reading, not the plain straight
}

TEST("T-STR-06") {
  const int L = rank_index('2');
  const RSet r = interp("H2 S6 D7 C8 D9", L);
  CHECK(has(r, Type::Straight, 4));
  CHECK(has(r, Type::Straight, 5));
  const RSet b = best("H2 S6 D7 C8 D9", L);
  CHECK(has(b, Type::Straight, 5));
  CHECK(!has(b, Type::Straight, 4));
}

TEST("T-STR-07") {
  CHECK(interpret(Hand::from_string("S3 S4 S5 S6 S7 S8"), rank_index('2')).empty());
}

TEST("T-STR-08") {
  CHECK(interpret(Hand::from_string("ST SJ SQ SK SB"), rank_index('2')).empty());
}

TEST("T-STR-09") {
  const int L = rank_index('7');
  const RSet r = interp("H7 S3 S4 S5 S6", L);
  CHECK(has(r, Type::Straight, 1));
  CHECK(has(r, Type::Straight, 2));
  CHECK(has(r, Type::StraightFlush, 1));
  CHECK(has(r, Type::StraightFlush, 2));
  const RSet b = best("H7 S3 S4 S5 S6", L);
  CHECK(has(b, Type::Straight, 2));
  CHECK(has(b, Type::StraightFlush, 2));
  CHECK(!has(b, Type::Straight, 1));
  CHECK(!has(b, Type::StraightFlush, 1));
}

TEST("T-STR-10") {
  const int L = rank_index('A');
  const RSet r = interp("HT HJ HQ HK HA", L);
  CHECK(has(r, Type::Straight, 8));
  CHECK(has(r, Type::Straight, 9));
  CHECK(has(r, Type::StraightFlush, 8));
  CHECK(has(r, Type::StraightFlush, 9));
}

// ---- Tubes and plates ----------------------------------------------------------

TEST("T-SEQ-01") {
  const int L = rank_index('7');
  const RSet r = interp("SQ DQ SK DK SA DA", L);
  CHECK(has(r, Type::Tube, 11));
  CHECK_EQ(r.size(), (size_t)1);
}

TEST("T-SEQ-02") {
  CHECK(interpret(Hand::from_string("SK DK SA DA S2 D2"), rank_index('7')).empty());
}

TEST("T-SEQ-03") {
  const int L = rank_index('7');
  const RSet r = interp("SA DA S2 D2 S3 D3", L);
  CHECK(has(r, Type::Tube, 0));
  CHECK_EQ(r.size(), (size_t)1);
}

TEST("T-SEQ-04") {
  const int L = rank_index('7');
  const RSet r1 = interp("SK DK CK SA DA CA", L);
  CHECK(has(r1, Type::Plate, 12));
  CHECK_EQ(r1.size(), (size_t)1);
  const RSet r2 = interp("SA DA CA S2 D2 C2", L);
  CHECK(has(r2, Type::Plate, 0));
  CHECK_EQ(r2.size(), (size_t)1);
}

TEST("T-SEQ-05") {
  const int L = rank_index('7');
  const RSet r = interp("H7 H7 S3 D3 S4 D4", L);
  CHECK(has(r, Type::Tube, 1));
  CHECK(has(r, Type::Tube, 2));
  CHECK(has(r, Type::Plate, 2));
  const RSet b = best("H7 H7 S3 D3 S4 D4", L);
  CHECK(has(b, Type::Tube, 2));
  CHECK(has(b, Type::Plate, 2));
  CHECK(!has(b, Type::Tube, 1));
}

// ---- Bombs ---------------------------------------------------------------------

namespace {
Hand rank_bomb(int rank, int count, int skip_suit = -1) {
  Hand h;
  int added = 0;
  for (int copy = 0; copy < 2 && added < count; ++copy) {
    for (int s = 0; s < kNumSuits && added < count; ++s) {
      if (s == skip_suit) continue;
      h.add(card_of(rank, s));
      ++added;
    }
  }
  return h;
}
}  // namespace

TEST("T-BOMB-01") {
  const int L = rank_index('7');
  Hand h;
  for (int s = 0; s < kNumSuits; ++s) { h.add(card_of(5, s)); h.add(card_of(5, s)); }
  const RSet r = to_set(interpret(h, L));
  CHECK(has(r, Type::Bomb, 12, 8));
}

TEST("T-BOMB-02") {
  const int L = rank_index('7');
  Hand h = rank_bomb(7, 8);  // eight natural 9s (rank index 7)
  h.add(wild_id(L));
  h.add(wild_id(L));
  const RSet r = to_set(interpret(h, L));
  CHECK(has(r, Type::Bomb, 6, 10));
  Hand h11 = h;
  h11.add(card_of(6, 0));  // an 11th card
  CHECK(interpret(h11, L).empty());
}

TEST("T-BOMB-03") {
  const int L = rank_index('7');
  const Hand b5 = rank_bomb(0, 5);   // 2s
  const Hand b4 = rank_bomb(12, 4);  // aces
  const Reading r5 = interpret(b5, L).front();
  const Reading r4 = interpret(b4, L).front();
  CHECK(beats_reading(r5.type, r5.key, r5.bomb_size, r4.type, r4.key, r4.bomb_size));
}

TEST("T-BOMB-04") {
  const int L = rank_index('7');
  const Reading sf0 = only("SA S2 S3 S4 S5", L);
  const Hand levelBomb5 = rank_bomb(rank_index('7'), 5, /*skip_suit=*/kHearts);
  const Reading lb5 = interpret(levelBomb5, L).front();
  CHECK(sf0.type == Type::StraightFlush);
  CHECK(beats_reading(sf0.type, sf0.key, sf0.bomb_size, lb5.type, lb5.key, lb5.bomb_size));

  const Hand bomb6 = rank_bomb(0, 6);  // 2s
  const Reading b6 = interpret(bomb6, L).front();
  const Reading sf9 = only("ST SJ SQ SK SA", L);
  CHECK(sf9.type == Type::StraightFlush);
  CHECK(beats_reading(b6.type, b6.key, b6.bomb_size, sf9.type, sf9.key, sf9.bomb_size));
}

TEST("T-BOMB-05") {
  const int L = rank_index('7');
  Hand jb;
  jb.add(kBJ); jb.add(kBJ); jb.add(kRJ); jb.add(kRJ);
  const Reading joker = interpret(jb, L).front();
  const Hand b10 = rank_bomb(6, 8);  // 8 natural cards of one rank
  Hand b10full = b10;
  b10full.add(wild_id(L));
  b10full.add(wild_id(L));
  const Reading ten = interpret(b10full, L).front();
  CHECK(beats_reading(joker.type, joker.key, joker.bomb_size, ten.type, ten.key, ten.bomb_size));
  CHECK(!beats_reading(ten.type, ten.key, ten.bomb_size, joker.type, joker.key, joker.bomb_size));
}

TEST("T-BOMB-06") {
  const int L = rank_index('7');
  CHECK(interpret(Hand::from_string("SB SB HR"), L).empty());
  CHECK(interpret(Hand::from_string("SB SB HR H7"), L).empty());
  const RSet r = interp("SB SB HR HR", L);
  CHECK(has(r, Type::JokerBomb, 0));
}

TEST("T-BOMB-07") {
  const int L = rank_index('7');
  const Hand b4 = rank_bomb(0, 4);  // 2s
  const Reading bomb = interpret(b4, L).front();
  const Reading fh = only("S7 D7 C7 SA DA", L);  // full house, triple 7 over pair A
  CHECK(fh.type == Type::FullHouse);
  CHECK(beats_reading(bomb.type, bomb.key, bomb.bomb_size, fh.type, fh.key, fh.bomb_size));
}

TEST("T-BOMB-08") {
  const int L = rank_index('7');
  const Reading sf = only("S3 S4 S5 S6 S7", L);
  CHECK(!beats_reading(sf.type, sf.key, sf.bomb_size, sf.type, sf.key, sf.bomb_size));
}

// ---- Mismatched types ------------------------------------------------------------

TEST("T-MIS-01") {
  CHECK(!beats_reading(Type::Pair, 5, 0, Type::Single, 3, 0));
  CHECK(!beats_reading(Type::Tube, 5, 0, Type::Plate, 3, 0));
  CHECK(!beats_reading(Type::Straight, 5, 0, Type::FullHouse, 3, 0));
}

// ---- Round bookkeeping ------------------------------------------------------------

TEST("T-RND-01") {
  {
    const std::array<int8_t, 4> order = {0, 2, 1, 3};
    const LevelGain g = level_gain(order);
    CHECK_EQ(g.team, 0);
    CHECK_EQ(g.gain, 3);
  }
  {
    const std::array<int8_t, 4> order = {0, 1, 2, 3};
    const LevelGain g = level_gain(order);
    CHECK_EQ(g.team, 0);
    CHECK_EQ(g.gain, 2);
  }
  {
    const std::array<int8_t, 4> order = {0, 1, 3, 2};
    const LevelGain g = level_gain(order);
    CHECK_EQ(g.team, 0);
    CHECK_EQ(g.gain, 1);
  }
  {
    const std::array<int8_t, 4> order = {1, 0, 3, 2};
    const LevelGain g = level_gain(order);
    CHECK_EQ(g.team, 1);
    CHECK_EQ(g.gain, 2);
  }
}

TEST("T-RND-02") {
  CHECK_EQ(promote(11, 3), 12);  // K + 3
  CHECK_EQ(promote(10, 3), 12);  // Q + 3
  CHECK_EQ(promote(12, 1), 12);  // A + 1
}

// ---- Abstract action ids (RULES.md 11.1) -------------------------------------------

TEST("abstract action count is 393") { CHECK_EQ(kNumAbstract, 393); }

TEST("abstract_id/abstract_action round trip over all 393 ids") {
  for (int id = 0; id < kNumAbstract; ++id) {
    Action a = abstract_action(id);
    if (a.type == Type::StraightFlush) {
      const int suit = (id - kAbstractStraightFlush) % 4;
      std::vector<CardId> cs;
      for (int r : straight_window(a.key)) cs.push_back(card_of(r, suit));
      a.cards = Hand::from_vector(cs);
    }
    const int back = abstract_id(a);
    CHECK_EQ(back, id);
  }
}

TEST("abstract action type counts match RULES.md 11.1") {
  int counts[(int)Type::kNumTypes] = {0};
  for (int id = 0; id < kNumAbstract; ++id) {
    counts[(int)abstract_action(id).type]++;
  }
  CHECK_EQ(counts[(int)Type::Pass], 1);
  CHECK_EQ(counts[(int)Type::Single], 15);
  CHECK_EQ(counts[(int)Type::Pair], 15);
  CHECK_EQ(counts[(int)Type::Triple], 13);
  CHECK_EQ(counts[(int)Type::FullHouse], 182);
  CHECK_EQ(counts[(int)Type::Straight], 10);
  CHECK_EQ(counts[(int)Type::Tube], 12);
  CHECK_EQ(counts[(int)Type::Plate], 13);
  CHECK_EQ(counts[(int)Type::Bomb], 91);
  CHECK_EQ(counts[(int)Type::StraightFlush], 40);
  CHECK_EQ(counts[(int)Type::JokerBomb], 1);
}
