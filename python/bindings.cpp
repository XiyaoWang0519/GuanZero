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
#include "gd/state.h"

namespace py = pybind11;
using namespace gd;

namespace {

Hand hand_from(const std::vector<int>& cs) {
  Hand h;
  for (int c : cs) h.add(static_cast<CardId>(c));
  return h;
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
py::array_t<T> view_of(const std::span<const T>& s) {
  // A read-only view that aliases the engine's own buffers, as DESIGN.md 5.3
  // requires. Valid until the next pending() or step().
  return py::array_t<T>({static_cast<py::ssize_t>(s.size())},
                        {static_cast<py::ssize_t>(sizeof(T))}, s.data(), py::cast(1));
}

}  // namespace

PYBIND11_MODULE(_gd_core, m) {
  m.doc() = "Guandan rules engine. See docs/PY_API.md.";
  m.attr("OBS_DIM") = kObsDim;
  m.attr("ACT_DIM") = kActDim;
  m.attr("NUM_ABSTRACT") = kNumAbstract;

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

  // ---- configuration ------------------------------------------------------
  py::enum_<Phase>(m, "Phase")
      .value("Deal", Phase::Deal).value("Tribute", Phase::Tribute)
      .value("BackTribute", Phase::BackTribute).value("Play", Phase::Play)
      .value("RoundEnd", Phase::RoundEnd).value("MatchEnd", Phase::MatchEnd);

  py::class_<RuleConfig>(m, "RuleConfig")
      .def(py::init<>())
      .def_static("house", &RuleConfig::house)
      .def_static("ogd", &RuleConfig::ogd)
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
            for (size_t i = 0; i < 4 && i < hs.size(); ++i) d.hands[i] = hand_from(hs[i]);
          })
      .def_property("level", [](const DealSpec& d) { return int(d.level); },
                    [](DealSpec& d, int v) { d.level = int8_t(v); })
      .def_property("team_levels",
          [](const DealSpec& d) { return std::vector<int>{d.team_levels[0], d.team_levels[1]}; },
          [](DealSpec& d, const std::vector<int>& v) {
            d.team_levels = {int8_t(v[0]), int8_t(v[1])};
          })
      .def_property("fails",
          [](const DealSpec& d) { return std::vector<int>{d.fails[0], d.fails[1]}; },
          [](DealSpec& d, const std::vector<int>& v) { d.fails = {int8_t(v[0]), int8_t(v[1])}; })
      .def_property("owner", [](const DealSpec& d) { return int(d.owner); },
                    [](DealSpec& d, int v) { d.owner = int8_t(v); })
      .def_property("leader", [](const DealSpec& d) { return int(d.leader); },
                    [](DealSpec& d, int v) { d.leader = int8_t(v); })
      .def_property("prev_order",
          [](const DealSpec& d) {
            return std::vector<int>{d.prev_order[0], d.prev_order[1],
                                    d.prev_order[2], d.prev_order[3]};
          },
          [](DealSpec& d, const std::vector<int>& v) {
            for (int i = 0; i < 4; ++i) d.prev_order[i] = int8_t(v[i]);
            d.has_prev = true;
          });

  py::class_<RoundResult>(m, "RoundResult")
      .def_property_readonly("order", [](const RoundResult& r) {
        return std::vector<int>{r.order[0], r.order[1], r.order[2], r.order[3]};
      })
      .def_property_readonly("num_finished_seats",
                             [](const RoundResult& r) { return count_finished(r.order); })
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
        const auto v = s.round.hands[seat].to_vector();
        return std::vector<int>(v.begin(), v.end());
      }, py::arg("seat"))
      .def("played", [](const MatchState& s, int seat) {
        const auto v = s.round.played[seat].to_vector();
        return std::vector<int>(v.begin(), v.end());
      }, py::arg("seat"))
      .def("hash", [](const MatchState& s) { return s.hash(); })
      .def("serialize", [](const MatchState& s) {
        return py::bytes(reinterpret_cast<const char*>(&s), sizeof(MatchState));
      })
      .def_static("deserialize", [](const py::bytes& b) {
        MatchState s;
        const std::string str = b;
        if (str.size() != sizeof(MatchState))
          throw std::runtime_error("bad MatchState blob");
        std::memcpy(&s, str.data(), sizeof(MatchState));
        return s;
      }, py::arg("blob"))
      .def("observation", [](const MatchState& s, int seat) {
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
      .def("end_round", [](const Engine& e, MatchState& s) {
        RoundResult r;
        e.end_round(s, r);
        return r;
      }, py::arg("state"))
      .def("encode_action", [](const Engine&, const Action& a, const MatchState& s, int seat) {
        py::array_t<float> out(kActDim);
        encode_action(a, s.round, seat, std::span<float>(out.mutable_data(), kActDim));
        return out;
      }, py::arg("action"), py::arg("state"), py::arg("seat"))
      .def("greedy", [](const Engine& e, const MatchState& s, uint64_t seed) {
        std::vector<Action> out;
        e.legal_actions(s, out);
        uint64_t rng = seed;
        return greedy_bot(s, out, rng);
      }, py::arg("state"), py::arg("seed") = 0);

  // ---- vectorized environment --------------------------------------------
  py::class_<DecisionBatch>(m, "DecisionBatch")
      .def_readonly("rows", &DecisionBatch::rows)
      .def_property_readonly("obs", [](const DecisionBatch& b) {
        py::array_t<float> a = view_of(b.obs);
        a.resize({b.rows, kObsDim});
        return a;
      })
      .def_property_readonly("cand", [](const DecisionBatch& b) {
        py::array_t<float> a = view_of(b.cand);
        a.resize({static_cast<py::ssize_t>(b.cand.size() / kActDim), py::ssize_t(kActDim)});
        return a;
      })
      .def_property_readonly("offsets", [](const DecisionBatch& b) { return view_of(b.offsets); })
      .def_property_readonly("env_id", [](const DecisionBatch& b) { return view_of(b.env_id); })
      .def_property_readonly("seat", [](const DecisionBatch& b) { return view_of(b.seat); })
      .def_property_readonly("phase", [](const DecisionBatch& b) { return view_of(b.phase); });

  py::class_<VecEnv>(m, "VecEnv")
      .def(py::init([](int num_envs, int num_threads, uint64_t seed,
                       RuleConfig rules, ActionConfig actions, bool encode) {
             EnvConfig cfg;
             cfg.rules = rules;
             cfg.actions = actions;
             cfg.encode = encode;
             return new VecEnv(num_envs, num_threads, cfg, seed);
           }),
           py::arg("num_envs"), py::arg("num_threads") = 1, py::arg("seed") = 0,
           py::arg("rules") = RuleConfig::house(), py::arg("actions") = ActionConfig{},
           py::arg("encode") = true)
      .def("reset", [](VecEnv& e, const std::vector<DealSpec>& deals) {
        e.reset(std::span<const DealSpec>(deals.data(), deals.size()));
      }, py::arg("deals") = std::vector<DealSpec>{})
      .def("pending", &VecEnv::pending)
      .def("step", [](VecEnv& e, py::array_t<int32_t, py::array::c_style | py::array::forcecast> ch) {
        e.step(std::span<const int32_t>(ch.data(), ch.size()));
      }, py::arg("choices"))
      .def("drain_finished_rounds", [](VecEnv& e) {
        const auto s = e.drain_finished_rounds();
        return std::vector<RoundResult>(s.begin(), s.end());
      })
      .def("fork", &VecEnv::fork, py::arg("env_id"), py::arg("copies"))
      .def_property_readonly("num_envs", &VecEnv::num_envs);
}
