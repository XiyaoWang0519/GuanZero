// Round trips and multiset properties for gd::Hand and card text (RULES.md 3).
#include <algorithm>
#include <random>
#include <vector>

#include "gd/cards.h"
#include "test_util.h"

using namespace gd;

TEST("card id/text round trip over all 54 ids") {
  for (int c = 0; c < kNumCardIds; ++c) {
    const std::string s = card_to_string(static_cast<CardId>(c));
    const int back = card_from_string(s);
    CHECK_EQ(back, c);
  }
  std::vector<CardId> all;
  for (int c = 0; c < kNumCardIds; ++c) all.push_back(static_cast<CardId>(c));
  const std::string joined = cards_to_string(all);
  const std::vector<CardId> back = cards_from_string(joined);
  CHECK(back == all);
}

TEST("card text edge cases") {
  CHECK_EQ(card_from_string("SB"), (int)kBJ);
  CHECK_EQ(card_from_string("HR"), (int)kRJ);
  CHECK(card_to_string(kBJ) == "SB");
  CHECK(card_to_string(kRJ) == "HR");
  CHECK_EQ(card_from_string(""), -1);
  CHECK_EQ(card_from_string("Z9"), -1);
  CHECK_EQ(card_from_string("S1"), -1);
}

TEST("Hand::to_vector/from_vector/from_string/to_string round trip") {
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> pick(0, kNumCardIds - 1);
  for (int trial = 0; trial < 200; ++trial) {
    Hand h;
    std::vector<CardId> added;
    const int n = pick(rng) % 20 + 1;
    for (int i = 0; i < n; ++i) {
      const CardId c = static_cast<CardId>(pick(rng));
      if (h.count(c) < 2) {
        h.add(c);
        added.push_back(c);
      }
    }
    std::vector<CardId> v1 = h.to_vector();
    std::vector<CardId> v2 = added;
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());
    CHECK(v1 == v2);

    const Hand h2 = Hand::from_vector(v1);
    CHECK(h2 == h);

    const std::string s = h.to_string();
    const Hand h3 = Hand::from_string(s);
    CHECK(h3 == h);
  }
}

TEST("Hand::add/remove/contains on random multisets") {
  std::mt19937 rng(999);
  std::uniform_int_distribution<int> pick(0, kNumCardIds - 1);
  for (int trial = 0; trial < 500; ++trial) {
    const CardId c = static_cast<CardId>(pick(rng));
    Hand h;
    CHECK_EQ(h.count(c), 0);
    h.add(c);
    CHECK_EQ(h.count(c), 1);
    CHECK(!h.empty());
    h.add(c);
    CHECK_EQ(h.count(c), 2);
    CHECK_EQ(h.size(), 2);

    Hand single;
    single.add(c);
    CHECK(h.contains(single));

    h.remove(c);
    CHECK_EQ(h.count(c), 1);
    h.remove(c);
    CHECK_EQ(h.count(c), 0);
    CHECK(h.empty());
  }
}

TEST("Hand::add_all/remove_all on random multisets") {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> pick(0, kNumCardIds - 1);
  for (int trial = 0; trial < 200; ++trial) {
    Hand h;
    const int n = pick(rng) % 15 + 1;
    for (int i = 0; i < n; ++i) {
      const CardId c = static_cast<CardId>(pick(rng));
      if (h.count(c) < 2) h.add(c);
    }
    const Hand original = h;

    // o is a random submultiset of h's remaining capacity, so add_all never
    // exceeds two copies of any card id.
    Hand o;
    for (int c = 0; c < kNumCardIds; ++c) {
      const int cap = 2 - h.count(static_cast<CardId>(c));
      const int take = cap > 0 ? (pick(rng) % (cap + 1)) : 0;
      for (int k = 0; k < take; ++k) o.add(static_cast<CardId>(c));
    }

    h.add_all(o);
    for (int c = 0; c < kNumCardIds; ++c) {
      CHECK_EQ(h.count(static_cast<CardId>(c)),
               original.count(static_cast<CardId>(c)) + o.count(static_cast<CardId>(c)));
    }
    CHECK(h.contains(o));
    CHECK(h.contains(original));

    h.remove_all(o);
    CHECK(h == original);
  }
}
