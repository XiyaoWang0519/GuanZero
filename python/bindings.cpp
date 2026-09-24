// Python bindings. The specification is docs/PY_API.md and the suites in
// tests/, which compare us with oracle/gd_reference.py shape for shape.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "gd/bots.h"
#include "gd/encoder.h"
#include "gd/env.h"
#include "gd/movegen.h"
#include "gd/rules.h"
#include "gd/search.h"
#include "gd/state.h"

namespace py = pybind11;
using namespace gd;

namespace {

void require_range(int value, int low, int high, const char* field) {
  if (value < low || value > high)
    throw py::value_error(std::string(field) + " outside valid range");
}

void require_size(size_t actual, size_t expected, const char* field) {
  if (actual != expected)
    throw py::value_error(std::string(field) + " has incorrect length");
}

Hand hand_from(const std::vector<int>& cs) {
  Hand h;
  for (int c : cs) {
    require_range(c, 0, kNumCardIds - 1, "card id");
    if (h.count(static_cast<CardId>(c)) >= 2)
      throw py::value_error("a hand cannot contain more than two copies of a card");
    h.add(static_cast<CardId>(c));
  }
  return h;
}

// Play-type name to its index in the engine's enum, for the style vector's
// per-type preference block.
int type_index(const std::string& name) {
  for (int i = 0; i < static_cast<int>(Type::kNumTypes); ++i)
    if (name == type_name(static_cast<Type>(i))) return i;
  throw py::value_error("unknown play type '" + name + "'");
}

// The oracle's key shape: a plain int, except Bomb which is (size, power).
py::object key_object(Type t, int key, int bomb_size) {
  if (t == Type::Bomb) return py::make_tuple(bomb_size, key);
  return py::int_(key);
}

py::set readings_to_set(const std::vector<Reading>& rs) {
  py::set out;
  for (const auto& r : rs)
    out.add(py::make_tuple(py::str(type_name(r.type)),
                           key_object(r.type, r.key, r.bomb_size)));
  return out;
}

// Turns an oracle-shaped (type, key) pair into an Action carrying no cards.
Action action_from_reading(const py::object& obj) {
  Action a;
  if (obj.is_none()) return a;   // leading
  const auto t = obj.cast<py::tuple>();
  const std::string name = t[0].cast<std::string>();
  for (int i = 0; i < static_cast<int>(Type::kNumTypes); ++i) {
    if (name == type_name(static_cast<Type>(i))) { a.type = static_cast<Type>(i); break; }
  }
  if (a.type == Type::Bomb) {
    const auto k = t[1].cast<py::tuple>();
    a.bomb_size = static_cast<int8_t>(k[0].cast<int>());
    a.key = static_cast<int8_t>(k[1].cast<int>());
  } else if (a.type != Type::Pass) {
    a.key = static_cast<int8_t>(t[1].cast<int>());
  }
  return a;
}

py::tuple action_triple(const Action& a) {
  const auto cs = a.cards.to_vector();
  py::tuple cards(cs.size());
  for (size_t i = 0; i < cs.size(); ++i) cards[i] = py::int_(int(cs[i]));
  return py::make_tuple(py::str(type_name(a.type)),
                        key_object(a.type, a.key, a.bomb_size), cards);
}

int count_finished(const std::array<int8_t, 4>& order) {
  int n = 0;
  for (int8_t s : order) n += (s >= 0) ? 1 : 0;
  return n;
}

template <typename T>
py::array_t<T> view_of(const std::span<const T>& s, const DecisionBatch& batch) {
  // A read-only view that aliases the engine's own buffers, as DESIGN.md 5.3
  // requires. Valid until the next pending() or step().
  auto a = py::array_t<T>({static_cast<py::ssize_t>(s.size())},
                         {static_cast<py::ssize_t>(sizeof(T))}, s.data(),
                         py::cast(&batch, py::return_value_policy::reference));
  a.attr("setflags")(false);
  return a;
}

}  // namespace

PYBIND11_MODULE(_gd_core, m) {
  m.doc() = "Guandan rules engine. See docs/PY_API.md.";
  m.attr("OBS_DIM") = kObsDim;
  m.attr("ACT_DIM") = kActDim;
  m.attr("NUM_ABSTRACT") = kNumAbstract;
  m.attr("STYLE_DIM") = StyleParams::kDim;
  m.attr("STYLE_BOMB_THRESHOLD") = StyleParams::kBombThreshold;
  m.attr("STYLE_TYPE_PREF") = StyleParams::kTypePref;
  m.attr("STYLE_FOLLOW_AGGRESSION") = StyleParams::kFollowAggression;
  m.attr("STYLE_LEAD_HIGH_BIAS") = StyleParams::kLeadHighBias;
  m.attr("STYLE_PARTNER_WEIGHT") = StyleParams::kPartnerWeight;
  m.attr("STYLE_TEMPERATURE") = StyleParams::kTemperature;
  // Layout constants, so that the golden tests pin the offsets from Python too.
  m.attr("ACT_CARDS1") = kActCards1;
  m.attr("ACT_CARDS2") = kActCards2;
  m.attr("ACT_TYPE") = kActType;
  m.attr("ACT_KEY") = kActKey;
  m.attr("ACT_BOMB_SIZE") = kActBombSize;
  m.attr("ACT_WILDS") = kActWilds;
  m.attr("ACT_TRIBUTE_FLAGS") = kActTributeFlags;
  m.attr("OBS_OWN_HAND") = kObsOwnHand;
  m.attr("OBS_UNSEEN") = kObsUnseen;
  m.attr("OBS_PLAYED") = kObsPlayed;
  m.attr("OBS_CARDS_LEFT") = kObsCardsLeft;
  m.attr("OBS_FINISH_STATUS") = kObsFinishStatus;
  m.attr("OBS_LEVELS") = kObsLevels;
  m.attr("OBS_WILD_HELD") = kObsWildHeld;
  m.attr("OBS_WILD_UNSEEN") = kObsWildUnseen;
  m.attr("OBS_WILD_FLAGS") = kObsWildFlags;
  m.attr("OBS_TRICK_TOP") = kObsTrickTop;
  m.attr("OBS_TRICK_HOLDER") = kObsTrickHolder;
  m.attr("OBS_TRICK_PASSES") = kObsTrickPasses;
  m.attr("OBS_TRICK_LEADING") = kObsTrickLeading;
  m.attr("OBS_LAST_ACTION") = kObsLastAction;
  m.attr("OBS_PHASE") = kObsPhase;
  m.attr("OBS_ROLES") = kObsRoles;
  m.attr("OBS_TRIBUTE") = kObsTribute;
  m.attr("OBS_KNOWN_HOLDINGS") = kObsKnownHoldings;

  // ---- configuration ------------------------------------------------------
  py::enum_<Phase>(m, "Phase")
      .value("Deal", Phase::Deal).value("Tribute", Phase::Tribute)
      .value("BackTribute", Phase::BackTribute).value("Play", Phase::Play)
      .value("RoundEnd", Phase::RoundEnd).value("MatchEnd", Phase::MatchEnd);

  py::enum_<TributeTie>(m, "TributeTie")
      .value("Upstream", TributeTie::Upstream)
      .value("LastFinisher", TributeTie::LastFinisher)
      .value("Downstream", TributeTie::Downstream);

  py::class_<RuleConfig>(m, "RuleConfig")
      .def(py::init<>())
      .def_static("house", &RuleConfig::house)
      .def_static("ogd", &RuleConfig::ogd)
      .def_readwrite("tribute_tie", &RuleConfig::tribute_tie)
      .def_readwrite("back_tribute_level_cards", &RuleConfig::back_tribute_level_cards)
      .def_readwrite("back_tribute_fallback", &RuleConfig::back_tribute_fallback)
      .def_readwrite("full_house_joker_pair", &RuleConfig::full_house_joker_pair)
      .def_readwrite("pass_a_requires_owner", &RuleConfig::pass_a_requires_owner)
      .def_readwrite("a_fail_on_loss", &RuleConfig::a_fail_on_loss)
      .def_readwrite("a_fail_limit", &RuleConfig::a_fail_limit)
      .def_readwrite("a_fail_reset_level", &RuleConfig::a_fail_reset_level)
      .def_readwrite("shuffle_limit", &RuleConfig::shuffle_limit)
      .def_readwrite("tribute_between_partners", &RuleConfig::tribute_between_partners)
      .def_readwrite("fixed_first_leader", &RuleConfig::fixed_first_leader);

  py::class_<ActionConfig>(m, "ActionConfig")
      .def(py::init<>())
      .def_static("full", &ActionConfig::full)
      .def_readwrite("prune_dominated_readings", &ActionConfig::prune_dominated_readings)
      .def_readwrite("suit_dedup", &ActionConfig::suit_dedup);

  py::class_<Action>(m, "Action")
      .def(py::init<>())
      .def_property_readonly("type", [](const Action& a) { return std::string(type_name(a.type)); })
      .def_readonly("key", &Action::key)
      .def_readonly("bomb_size", &Action::bomb_size)
      .def_readonly("wilds", &Action::wilds)
      .def_property_readonly("cards", [](const Action& a) {
        const auto v = a.cards.to_vector();
        return std::vector<int>(v.begin(), v.end());
      })
      .def_property_readonly("abstract_id", [](const Action& a) { return abstract_id(a); })
      .def_property_readonly("is_pass", &Action::is_pass)
      .def("as_tuple", &action_triple)
      .def("__repr__", &Action::to_string);

  py::class_<DealSpec>(m, "DealSpec")
      .def(py::init<>())
      .def_property("hands",
          [](const DealSpec& d) {
            std::vector<std::vector<int>> out;
            for (const auto& h : d.hands) {
              const auto v = h.to_vector();
              out.emplace_back(v.begin(), v.end());
            }
            return out;
          },
          [](DealSpec& d, const std::vector<std::vector<int>>& hs) {
            require_size(hs.size(), 4, "hands");
            std::array<Hand, 4> hands;
            for (size_t i = 0; i < 4; ++i) hands[i] = hand_from(hs[i]);
            d.hands = hands;
          })
      .def_property("level", [](const DealSpec& d) { return int(d.level); },
                    [](DealSpec& d, int v) { require_range(v, 0, 12, "level"); d.level = int8_t(v); })
      .def_property("team_levels",
          [](const DealSpec& d) { return std::vector<int>{d.team_levels[0], d.team_levels[1]}; },
          [](DealSpec& d, const std::vector<int>& v) {
            require_size(v.size(), 2, "team_levels");
            for (int x : v) require_range(x, 0, 12, "team level");
            d.team_levels = {int8_t(v[0]), int8_t(v[1])};
          })
      .def_property("fails",
          [](const DealSpec& d) { return std::vector<int>{d.fails[0], d.fails[1]}; },
          [](DealSpec& d, const std::vector<int>& v) {
            require_size(v.size(), 2, "fails");
            for (int x : v) require_range(x, 0, 127, "fails");
            d.fails = {int8_t(v[0]), int8_t(v[1])};
          })
      .def_property("owner", [](const DealSpec& d) { return int(d.owner); },
                    [](DealSpec& d, int v) { require_range(v, -1, 1, "owner"); d.owner = int8_t(v); })
      .def_property("leader", [](const DealSpec& d) { return int(d.leader); },
                    [](DealSpec& d, int v) { require_range(v, -1, 3, "leader"); d.leader = int8_t(v); })
      .def_property("prev_order",
          [](const DealSpec& d) {
            return std::vector<int>{d.prev_order[0], d.prev_order[1],
                                    d.prev_order[2], d.prev_order[3]};
          },
          [](DealSpec& d, const std::vector<int>& v) {
            require_size(v.size(), 4, "prev_order");
            if (std::set<int>(v.begin(), v.end()) != std::set<int>{0, 1, 2, 3})
              throw py::value_error("prev_order must be a permutation of seats 0..3");
            for (int i = 0; i < 4; ++i) d.prev_order[i] = int8_t(v[i]);
            d.has_prev = true;
          });

  py::class_<RoundResult>(m, "RoundResult")
      .def_readonly("env_id", &RoundResult::env_id)
      .def_readonly("match_id", &RoundResult::match_id)
      .def_readonly("round_index", &RoundResult::round_index)
      .def_property_readonly("order", [](const RoundResult& r) {
        return std::vector<int>{r.order[0], r.order[1], r.order[2], r.order[3]};
      })
      .def_property_readonly("num_finished_seats",
                             [](const RoundResult& r) { return int(r.num_out); })
      .def_property_readonly("winning_team", [](const RoundResult& r) { return int(r.winning_team); })
      .def_property_readonly("gain", [](const RoundResult& r) { return int(r.gain); })
      .def_property_readonly("levels", [](const RoundResult& r) {
        return std::vector<int>{r.levels[0], r.levels[1]};
      })
      .def_property_readonly("fails", [](const RoundResult& r) {
        return std::vector<int>{r.fails[0], r.fails[1]};
      })
      .def_property_readonly("next_owner", [](const RoundResult& r) { return int(r.next_owner); })
      .def_property_readonly("match_winner", [](const RoundResult& r) { return int(r.match_winner); })
      .def_property_readonly("round_level", [](const RoundResult& r) { return int(r.round_level); })
      .def_property_readonly("seat_return", [](const RoundResult& r) {
        return std::vector<int>{r.seat_return[0], r.seat_return[1],
                                r.seat_return[2], r.seat_return[3]};
      });

  py::class_<MatchState>(m, "MatchState")
      .def(py::init<>())
      .def_property_readonly("levels", [](const MatchState& s) {
        return std::vector<int>{s.levels[0], s.levels[1]};
      })
      .def_property_readonly("fails", [](const MatchState& s) {
        return std::vector<int>{s.fails[0], s.fails[1]};
      })
      .def_property_readonly("owner", [](const MatchState& s) { return int(s.owner); })
      .def_property_readonly("round_index", [](const MatchState& s) { return int(s.round_index); })
      .def_property_readonly("winner", [](const MatchState& s) { return int(s.winner); })
      .def_property_readonly("phase", [](const MatchState& s) { return s.round.phase; })
      .def_property_readonly("level", [](const MatchState& s) { return int(s.round.level); })
      .def_property_readonly("to_move", [](const MatchState& s) { return int(s.round.to_move); })
      .def_property_readonly("top_is_open",
                             [](const MatchState& s) { return s.round.top.is_pass(); })
      .def_property_readonly("order", [](const MatchState& s) {
        return std::vector<int>{s.round.order[0], s.round.order[1],
                                s.round.order[2], s.round.order[3]};
      })
      .def("hand", [](const MatchState& s, int seat) {
        require_range(seat, 0, 3, "seat");
        const auto v = s.round.hands[seat].to_vector();
        return std::vector<int>(v.begin(), v.end());
      }, py::arg("seat"))
      .def("played", [](const MatchState& s, int seat) {
        require_range(seat, 0, 3, "seat");
        const auto v = s.round.played[seat].to_vector();
        return std::vector<int>(v.begin(), v.end());
      }, py::arg("seat"))
      .def("hash", [](const MatchState& s) { return s.hash(); })
      .def("serialize", [](const MatchState& s) {
        return py::bytes(reinterpret_cast<const char*>(&s), sizeof(MatchState));
      })
      .def("determinize_uniform", &determinize_uniform,
           py::arg("observer"), py::arg("seed"))
      .def_static("deserialize", [](const py::bytes& b) {
        MatchState s;
        const std::string str = b;
        if (str.size() != sizeof(MatchState))
          throw std::runtime_error("bad MatchState blob");
        std::memcpy(&s, str.data(), sizeof(MatchState));
        return s;
      }, py::arg("blob"))
      .def("observation", [](const MatchState& s, int seat) {
        require_range(seat, 0, 3, "seat");
        py::array_t<float> out(kObsDim);
        encode_observation(s, seat, std::span<float>(out.mutable_data(), kObsDim));
        return out;
      }, py::arg("seat"));

  py::class_<Engine>(m, "Engine")
      .def(py::init<RuleConfig, ActionConfig>(),
           py::arg("rules") = RuleConfig::house(), py::arg("actions") = ActionConfig{})
      .def("new_match", [](const Engine& e, MatchState& s, uint64_t seed) {
        e.new_match(s, seed);
      }, py::arg("state"), py::arg("seed"))
      .def("set_deal", &Engine::set_deal, py::arg("state"), py::arg("deal"))
      .def("begin_round", &Engine::begin_round, py::arg("state"))
      .def("legal_actions", [](const Engine& e, const MatchState& s) {
        std::vector<Action> out;
        e.legal_actions(s, out);
        return out;
      }, py::arg("state"))
      .def("apply", &Engine::apply, py::arg("state"), py::arg("action"))
      .def("needs_decision", &Engine::needs_decision, py::arg("state"))
      .def_property("auto_pass", &Engine::auto_pass, &Engine::set_auto_pass)
      .def("end_round", [](const Engine& e, MatchState& s) {
        RoundResult r;
        e.end_round(s, r);
        return r;
      }, py::arg("state"))
      .def("encode_action", [](const Engine&, const Action& a, const MatchState& s, int seat) {
        require_range(seat, 0, 3, "seat");
        py::array_t<float> out(kActDim);
        encode_action(a, s.round, seat, std::span<float>(out.mutable_data(), kActDim));
        return out;
      }, py::arg("action"), py::arg("state"), py::arg("seat"))
      .def("greedy", [](const Engine& e, const MatchState& s, uint64_t seed) {
        std::vector<Action> out;
        e.legal_actions(s, out);
        uint64_t rng = seed;
        return greedy_bot(s, out, rng);
      }, py::arg("state"), py::arg("seed") = 0)
      .def("styled", [](const Engine& e, const MatchState& s,
                        py::array_t<float, py::array::c_style | py::array::forcecast> style,
                        uint64_t seed) {
        require_size(size_t(style.size()), size_t(StyleParams::kDim), "style vector");
        StyleParams p;
        std::memcpy(p.v.data(), style.data(), sizeof(float) * StyleParams::kDim);
        std::vector<Action> out;
        e.legal_actions(s, out);
        uint64_t rng = seed;
        return styled_bot(s, out, p, rng);
      }, py::arg("state"), py::arg("style"), py::arg("seed") = 0);


  // ---- styled bot ---------------------------------------------------------
  py::class_<StyleParams>(m, "StyleParams")
      .def(py::init<>())
      .def_static("neutral", &StyleParams::neutral)
      .def_static("from_array",
                  [](py::array_t<float, py::array::c_style | py::array::forcecast> a) {
                    require_size(size_t(a.size()), size_t(StyleParams::kDim), "style vector");
                    StyleParams p;
                    std::memcpy(p.v.data(), a.data(), sizeof(float) * StyleParams::kDim);
                    return p;
                  }, py::arg("values"))
      .def("to_array", [](const StyleParams& p) {
        py::array_t<float> a(StyleParams::kDim);
        std::memcpy(a.mutable_data(), p.v.data(), sizeof(float) * StyleParams::kDim);
        return a;
      })
      // A writable float32 view of the style's own storage, so that callers can
      // sample styles in place. Valid while the StyleParams object lives.
      .def_property_readonly("v", [](StyleParams& p) {
        return py::array_t<float>({py::ssize_t(StyleParams::kDim)},
                                  {py::ssize_t(sizeof(float))}, p.v.data(),
                                  py::cast(&p, py::return_value_policy::reference));
      })
      .def_property_readonly_static("DIM", [](py::object) { return StyleParams::kDim; })
      .def_property("bomb_threshold", &StyleParams::bomb_threshold,
                    [](StyleParams& p, float x) { p.v[StyleParams::kBombThreshold] = x; })
      .def_property("follow_aggression", &StyleParams::follow_aggression,
                    [](StyleParams& p, float x) { p.v[StyleParams::kFollowAggression] = x; })
      .def_property("lead_high_bias", &StyleParams::lead_high_bias,
                    [](StyleParams& p, float x) { p.v[StyleParams::kLeadHighBias] = x; })
      .def_property("partner_weight", &StyleParams::partner_weight,
                    [](StyleParams& p, float x) { p.v[StyleParams::kPartnerWeight] = x; })
      .def_property("temperature", &StyleParams::temperature,
                    [](StyleParams& p, float x) { p.v[StyleParams::kTemperature] = x; })
      .def("type_pref", [](const StyleParams& p, const std::string& name) {
        return p.v[StyleParams::kTypePref + type_index(name)];
      }, py::arg("type"))
      .def("set_type_pref", [](StyleParams& p, const std::string& name, float x) {
        p.v[StyleParams::kTypePref + type_index(name)] = x;
      }, py::arg("type"), py::arg("value"))
      .def("__len__", [](const StyleParams&) { return StyleParams::kDim; });

  // ---- cards and orderings ------------------------------------------------
  m.def("card_id", &card_from_string, py::arg("text"));
  m.def("card_str", [](int c) { return card_to_string(static_cast<CardId>(c)); }, py::arg("card"));
  m.def("cards", [](const std::string& s) {
    const auto v = cards_from_string(s);
    return std::vector<int>(v.begin(), v.end());
  }, py::arg("text"));
  m.def("power", &power, py::arg("rank"), py::arg("level"));
  m.def("abstract_action_count", [] { return kNumAbstract; });

  // ---- readings -----------------------------------------------------------
  m.def("interpret", [](const std::vector<int>& cs, int level) {
    return readings_to_set(interpret(hand_from(cs), level));
  }, py::arg("cards"), py::arg("level"));
  m.def("best_readings", [](const std::vector<int>& cs, int level) {
    return readings_to_set(best_readings(hand_from(cs), level));
  }, py::arg("cards"), py::arg("level"));
  m.def("beats", [](const py::object& cand, const py::object& top) {
    const Action c = action_from_reading(cand);
    if (top.is_none()) return c.type != Type::Pass;
    return beats(c, action_from_reading(top));
  }, py::arg("cand"), py::arg("top"));

  m.def("abstract_id", [](const py::object& obj) {
    if (py::isinstance<Action>(obj)) return abstract_id(obj.cast<const Action&>());
    const auto t = obj.cast<py::tuple>();
    Action a = action_from_reading(py::make_tuple(t[0], t[1]));
    if (t.size() > 2) {
      for (const auto& c : t[2].cast<py::tuple>()) a.cards.add(static_cast<CardId>(c.cast<int>()));
    }
    return abstract_id(a);
  }, py::arg("action"));

  // ---- move generation ----------------------------------------------------
  m.def("legal_actions", [](const std::vector<int>& hand, int level,
                            const py::object& top, bool canonical) {
    ActionConfig acfg = canonical ? ActionConfig{} : ActionConfig::full();
    std::vector<Action> out;
    generate_moves(hand_from(hand), level, action_from_reading(top), acfg,
                   RuleConfig::house(), out);
    py::set result;
    for (const auto& a : out) result.add(action_triple(a));
    return result;
  }, py::arg("hand"), py::arg("level"), py::arg("top"), py::arg("canonical") = false);

  m.def("is_legal", [](const std::vector<int>& hand, int level,
                       const py::object& top, const py::object& action) {
    Action a;
    if (py::isinstance<Action>(action)) {
      a = action.cast<Action>();
    } else {
      const auto t = action.cast<py::tuple>();
      a = action_from_reading(py::make_tuple(t[0], t[1]));
      if (t.size() > 2)
        for (const auto& c : t[2].cast<py::tuple>())
          a.cards.add(static_cast<CardId>(c.cast<int>()));
    }
    return is_legal(hand_from(hand), level, action_from_reading(top), a,
                    RuleConfig::house());
  }, py::arg("hand"), py::arg("level"), py::arg("top"), py::arg("action"));

  m.def("tribute_choices", [](const std::vector<int>& hand, int level) {
    std::vector<Action> out;
    generate_tribute(hand_from(hand), level, RuleConfig::house(), out);
    std::set<int> ids;
    for (const auto& a : out) ids.insert(int(a.cards.to_vector().front()));
    return ids;
  }, py::arg("hand"), py::arg("level"));
  m.def("back_tribute_choices", [](const std::vector<int>& hand, int level) {
    std::vector<Action> out;
    generate_back_tribute(hand_from(hand), level, RuleConfig::house(), out);
    std::set<int> ids;
    for (const auto& a : out) ids.insert(int(a.cards.to_vector().front()));
    return ids;
  }, py::arg("hand"), py::arg("level"));

  // ---- round bookkeeping --------------------------------------------------
  m.def("level_gain", [](const std::vector<int>& order) {
    std::array<int8_t, 4> o{};
    for (int i = 0; i < 4; ++i) o[i] = static_cast<int8_t>(order[i]);
    const LevelGain g = level_gain(o);
    return py::make_tuple(g.team, g.gain);
  }, py::arg("order"));
  m.def("promote", &promote, py::arg("level"), py::arg("gain"));
  m.def("end_of_round", [](const std::vector<int>& levels, const std::vector<int>& fails,
                           const py::object& owner, int round_level,
                           const std::vector<int>& order, const RuleConfig& rules) {
    std::array<int8_t, 2> lv{int8_t(levels[0]), int8_t(levels[1])};
    std::array<int8_t, 2> fl{int8_t(fails[0]), int8_t(fails[1])};
    std::array<int8_t, 4> o{};
    for (int i = 0; i < 4; ++i) o[i] = static_cast<int8_t>(order[i]);
    const EndOfRound r = end_of_round(lv, fl, owner.is_none() ? -1 : owner.cast<int>(),
                                      round_level, o, rules);
    py::object winner = r.match_winner < 0 ? py::none()
                                           : py::object(py::int_(int(r.match_winner)));
    return py::make_tuple(std::vector<int>{r.levels[0], r.levels[1]},
                          std::vector<int>{r.fails[0], r.fails[1]},
                          int(r.next_owner), winner);
  }, py::arg("levels"), py::arg("fails"), py::arg("owner"), py::arg("round_level"),
     py::arg("order"), py::arg("rules") = RuleConfig::house());

  // ---- vectorized environment --------------------------------------------
  py::class_<DecisionBatch>(m, "DecisionBatch")
      .def_readonly("rows", &DecisionBatch::rows)
      .def_property_readonly("obs", [](const DecisionBatch& b) {
        py::array_t<float> a = view_of(b.obs, b);
        a.resize({static_cast<int>(b.obs.size() / kObsDim), kObsDim});
        return a;
      })
      .def_property_readonly("cand", [](const DecisionBatch& b) {
        py::array_t<float> a = view_of(b.cand, b);
        a.resize({static_cast<py::ssize_t>(b.cand.size() / kActDim), py::ssize_t(kActDim)});
        return a;
      })
      .def_property_readonly("offsets", [](const DecisionBatch& b) { return view_of(b.offsets, b); })
      .def_property_readonly("env_id", [](const DecisionBatch& b) { return view_of(b.env_id, b); })
      .def_property_readonly("seat", [](const DecisionBatch& b) { return view_of(b.seat, b); })
      .def_property_readonly("phase", [](const DecisionBatch& b) { return view_of(b.phase, b); })
      .def_property_readonly("round_index", [](const DecisionBatch& b) { return view_of(b.round_index, b); })
      .def_property_readonly("match_id", [](const DecisionBatch& b) { return view_of(b.match_id, b); })
      .def_property_readonly("greedy_choice", [](const DecisionBatch& b) { return view_of(b.greedy_choice, b); })
      .def_property_readonly("styled_choice", [](const DecisionBatch& b) { return view_of(b.styled_choice, b); })
      .def_property_readonly("hidden_counts", [](const DecisionBatch& b) {
        auto a = view_of(b.hidden_counts, b);
        a.resize({b.rows, 3, kNumCardIds});
        return a;
      });

  py::class_<PublicActionEvent>(m, "PublicActionEvent")
      .def_readonly("env_id", &PublicActionEvent::env_id)
      .def_readonly("match_id", &PublicActionEvent::match_id)
      .def_readonly("round_index", &PublicActionEvent::round_index)
      .def_readonly("step", &PublicActionEvent::step)
      .def_readonly("seat", &PublicActionEvent::seat)
      .def_readonly("phase", &PublicActionEvent::phase)
      .def_readonly("action", &PublicActionEvent::action)
      .def_readonly("forced", &PublicActionEvent::forced)
      .def_readonly("cards_left", &PublicActionEvent::cards_left)
      .def_property_readonly("encoded_action", [](const PublicActionEvent& event) {
        py::array_t<float> a(kActDim);
        std::memcpy(a.mutable_data(), event.encoded_action.data(), sizeof(float) * kActDim);
        return a;
      });

  py::class_<VecEnv>(m, "VecEnv")
      .def(py::init([](int num_envs, int num_threads, uint64_t seed,
                       RuleConfig rules, ActionConfig actions, bool encode,
                       bool log_public_actions, int log_env_limit) {
             EnvConfig cfg;
             cfg.rules = rules;
             cfg.actions = actions;
             cfg.encode = encode;
             cfg.log_public_actions = log_public_actions;
             cfg.log_env_limit = log_env_limit;
             return new VecEnv(num_envs, num_threads, cfg, seed);
           }),
           py::arg("num_envs"), py::arg("num_threads") = 1, py::arg("seed") = 0,
           py::arg("rules") = RuleConfig::house(), py::arg("actions") = ActionConfig{},
           py::arg("encode") = true, py::arg("log_public_actions") = false,
           py::arg("log_env_limit") = -1)
      .def("reset", [](VecEnv& e, const std::vector<DealSpec>& deals,
                       const std::vector<uint64_t>& match_seeds) {
        e.reset(std::span<const DealSpec>(deals.data(), deals.size()), match_seeds);
      }, py::arg("deals") = std::vector<DealSpec>{},
         py::arg("match_seeds") = std::vector<uint64_t>{})
      .def("pending", &VecEnv::pending, py::keep_alive<0, 1>())
      .def("step", [](VecEnv& e, py::array_t<int32_t, py::array::c_style> ch) {
        if (ch.ndim() != 1) throw py::value_error("choices must be a one-dimensional int32 array");
        e.step(std::span<const int32_t>(ch.data(), ch.size()));
      }, py::arg("choices").noconvert())
      .def("drain_finished_rounds", [](VecEnv& e) {
        const auto s = e.drain_finished_rounds();
        return std::vector<RoundResult>(s.begin(), s.end());
      })
      .def("drain_public_actions", [](VecEnv& e) {
        const auto events = e.drain_public_actions();
        return std::vector<PublicActionEvent>(events.begin(), events.end());
      })
      .def("set_styles", [](VecEnv& e, py::array_t<float, py::array::c_style> styles) {
        if (styles.ndim() != 3 || styles.shape(1) != 4 ||
            styles.shape(2) != StyleParams::kDim)
          throw py::value_error("styles must be a float32 array [num_envs, 4, STYLE_DIM]");
        if (styles.shape(0) != e.num_envs())
          throw py::value_error("styles must have one row per environment");
        e.set_styles(std::span<const float>(styles.data(), size_t(styles.size())));
      }, py::arg("styles").noconvert())
      .def("clear_styles", &VecEnv::clear_styles)
      .def("row_actions", [](const VecEnv& e, int row) {
        const auto& actions = e.row_actions(row);
        return std::vector<Action>(actions.begin(), actions.end());
      }, py::arg("row"))
      .def("fork", &VecEnv::fork, py::arg("env_id"), py::arg("copies"))
      .def_property_readonly("num_envs", &VecEnv::num_envs);
}
