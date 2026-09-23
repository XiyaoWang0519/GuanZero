// Round and match state machine. The contract is docs/STATE_SPEC.md.
#include "gd/state.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "gd/movegen.h"
#include "gd/rules.h"

namespace gd {
namespace {

uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

void mix(uint64_t& h, uint64_t x) {
  h ^= x + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
}

constexpr int partner_of(int s) { return (s + 2) % 4; }
constexpr int team_of(int s) { return s % 2; }

void deal(RoundState& r, uint64_t& rng) {
  std::array<CardId, kDeckSize> deck{};
  int n = 0;
  for (int c = 0; c < kNumCardIds; ++c) {
    deck[n++] = static_cast<CardId>(c);
    deck[n++] = static_cast<CardId>(c);
  }
  for (int i = kDeckSize - 1; i > 0; --i) {
    const int j = static_cast<int>(splitmix64(rng) % static_cast<uint64_t>(i + 1));
    std::swap(deck[i], deck[j]);
  }
  for (int s = 0; s < 4; ++s) r.hands[s].clear();
  for (int i = 0; i < kDeckSize; ++i) r.hands[i / kHandSize].add(deck[i]);
}

}  // namespace

uint64_t RoundState::hash() const {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int s = 0; s < 4; ++s) {
    mix(h, hands[s].has1); mix(h, hands[s].has2);
    mix(h, played[s].has1); mix(h, played[s].has2);
    mix(h, static_cast<uint64_t>(finish_pos[s] + 2));
    mix(h, static_cast<uint64_t>(order[s] + 2));
    mix(h, last_action[s].cards.has1);
    mix(h, last_action[s].cards.has2);
    mix(h, static_cast<uint64_t>(last_action[s].type));
    mix(h, static_cast<uint64_t>(last_action[s].key + 1));
    mix(h, static_cast<uint64_t>(has_acted[s] ? 1 : 0));
  }
  mix(h, static_cast<uint64_t>(level));
  mix(h, static_cast<uint64_t>(to_move));
  mix(h, static_cast<uint64_t>(holder + 2));
  mix(h, top.cards.has1); mix(h, top.cards.has2);
  mix(h, static_cast<uint64_t>(top.type));
  mix(h, static_cast<uint64_t>(top.key + 1));
  mix(h, static_cast<uint64_t>(top.bomb_size));
  mix(h, static_cast<uint64_t>(passes));
  mix(h, static_cast<uint64_t>(num_finished));
  mix(h, static_cast<uint64_t>(num_out));
  mix(h, static_cast<uint64_t>(phase));
  mix(h, static_cast<uint64_t>(anti_tribute ? 1 : 0));
  mix(h, static_cast<uint64_t>(num_tribute_moves));
  for (int i = 0; i < num_tribute_moves; ++i) {
    mix(h, static_cast<uint64_t>(tribute_moves[i].payer + 2));
    mix(h, static_cast<uint64_t>(tribute_moves[i].receiver + 2));
    mix(h, static_cast<uint64_t>(tribute_moves[i].card));
    mix(h, static_cast<uint64_t>(tribute_moves[i].back ? 1 : 0));
  }
  for (int i = 0; i < 2; ++i) {
    mix(h, static_cast<uint64_t>(tribute_payers[i] + 2));
    mix(h, static_cast<uint64_t>(tribute_receivers[i] + 2));
    mix(h, static_cast<uint64_t>(tribute_cards[i] + 2));
  }
  mix(h, static_cast<uint64_t>(tribute_step));
  mix(h, static_cast<uint64_t>(first_leader + 2));
  return h;
}

uint64_t MatchState::hash() const {
  uint64_t h = round.hash();
  for (int t = 0; t < 2; ++t) {
    mix(h, static_cast<uint64_t>(levels[t]));
    mix(h, static_cast<uint64_t>(fails[t]));
  }
  mix(h, static_cast<uint64_t>(owner + 2));
  mix(h, static_cast<uint64_t>(round_index));
  mix(h, static_cast<uint64_t>(winner + 2));
  for (int s = 0; s < 4; ++s) mix(h, static_cast<uint64_t>(prev_order[s] + 2));
  mix(h, static_cast<uint64_t>(has_prev ? 1 : 0));
  return h;
}

EndOfRound end_of_round(const std::array<int8_t, 2>& levels,
                        const std::array<int8_t, 2>& fails, int owner,
                        int round_level, const std::array<int8_t, 4>& order,
                        const RuleConfig& rules) {
  EndOfRound r;
  r.levels = levels;
  r.fails = fails;
  const LevelGain g = level_gain(order);
  const int partner = partner_of(order[0]);
  bool partner_is_dweller = false;
  for (int i = 0; i < 4; ++i) if (order[i] == partner) partner_is_dweller = (i == 3);

  r.match_winner = -1;
  const bool a_round = owner >= 0 && round_level == 12 && levels[owner] == 12;
  if (a_round) {
    if (g.team == owner && !partner_is_dweller) {
      r.match_winner = static_cast<int8_t>(owner);
    } else if (g.team == owner || rules.a_fail_on_loss) {
      // The owner failed its attempt. Whether a plain loss counts is a profile
      // decision (RULES.md 13.1).
      // With no reset (house, a_fail_limit = 0) the count is unbounded, so it
      // saturates at the storage limit instead of wrapping.
      if (r.fails[owner] < std::numeric_limits<int8_t>::max())
        r.fails[owner] = static_cast<int8_t>(r.fails[owner] + 1);
    }
  }
  if (r.match_winner < 0) {
    r.levels[g.team] = static_cast<int8_t>(promote(r.levels[g.team], g.gain));
    if (owner >= 0 && rules.a_fail_limit && r.fails[owner] >= rules.a_fail_limit) {
      r.levels[owner] = static_cast<int8_t>(rules.a_fail_reset_level);
      r.fails[owner] = 0;
    }
  }
  r.next_owner = static_cast<int8_t>(g.team);
  return r;
}

TributeRouting double_tribute(int banker, int seat_a, int power_a,
                              [[maybe_unused]] int seat_b, int power_b,
                              const RuleConfig& rules) {
  const int down = (banker + 1) % 4;
  const int up = (banker + 3) % 4;
  const int p_down = seat_a == down ? power_a : power_b;
  const int p_up = seat_a == up ? power_a : power_b;

  int to_banker, to_follower;
  if (rules.tribute_pairing == TributePairing::SeatGeometry) {
    to_banker = down;
    to_follower = up;
  } else if (p_down > p_up) {
    to_banker = down; to_follower = up;
  } else if (p_up > p_down) {
    to_banker = up; to_follower = down;
  } else if (rules.tribute_tie == TributeTie::Downstream) {
    to_banker = down; to_follower = up;          // tie, clockwise (official)
  } else {
    to_banker = up; to_follower = down;          // tie, Upstream or LastFinisher
  }
  // The payer of the higher card leads under both profiles. On a tie the seat
  // that pays the Banker leads; apply_tribute overrides it for LastFinisher.
  int leader;
  if (p_down > p_up) leader = down;
  else if (p_up > p_down) leader = up;
  else leader = rules.tribute_tie == TributeTie::Downstream ? down : up;
  return {to_banker, to_follower, leader};
}

// ---------------------------------------------------------------------------

void Engine::new_match(MatchState& m, uint64_t seed) const {
  m = MatchState{};
  m.rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
  m.levels = {0, 0};
  m.fails = {0, 0};
  m.owner = -1;
  m.round_index = 0;
  m.winner = -1;
  m.has_prev = false;
  m.round = RoundState{};
  m.round.level = 0;
  deal(m.round, m.rng);
  const int leader = rules_.first_leader == FirstLeader::Fixed
                         ? rules_.fixed_first_leader
                         : static_cast<int>(splitmix64(m.rng) % 4);
  m.round.first_leader = static_cast<int8_t>(leader);
  m.round.to_move = static_cast<int8_t>(leader);
  m.round.phase = Phase::Play;
  skip_forced(m);
}

void Engine::set_deal(MatchState& m, const DealSpec& d) const {
  m = MatchState{};
  m.rng = 0x243F6A8885A308D3ULL;
  m.levels = d.team_levels;
  m.fails = d.fails;
  m.owner = d.owner;
  m.winner = -1;
  m.round = RoundState{};
  m.round.level = d.level;
  m.round.hands = d.hands;
  m.prev_order = d.prev_order;
  m.has_prev = d.has_prev;
  if (d.has_prev) {
    m.round_index = 1;
    open_tribute(m);
  } else {
    m.round_index = 0;
    const int leader = d.leader >= 0 ? d.leader : 0;
    m.round.first_leader = static_cast<int8_t>(leader);
    m.round.to_move = static_cast<int8_t>(leader);
    m.round.phase = Phase::Play;
  }
  skip_forced(m);
}

void Engine::begin_round(MatchState& m) const {
  RoundState fresh{};
  fresh.level = static_cast<int8_t>(m.owner >= 0 ? m.levels[m.owner] : 0);
  m.round = fresh;
  deal(m.round, m.rng);
  ++m.round_index;
  if (m.has_prev) {
    open_tribute(m);
  } else {
    const int leader = static_cast<int>(splitmix64(m.rng) % 4);
    m.round.first_leader = static_cast<int8_t>(leader);
    m.round.to_move = static_cast<int8_t>(leader);
    m.round.phase = Phase::Play;
  }
  skip_forced(m);
}

// Anti-tribute and the tribute schedule (RULES.md 9).
void Engine::open_tribute(MatchState& m) const {
  RoundState& r = m.round;
  const int banker = m.prev_order[0];
  const int follower = m.prev_order[1];
  const int third = m.prev_order[2];
  const int dweller = m.prev_order[3];
  const bool double_win = partner_of(banker) == follower;

  for (int s = 0; s < 4; ++s)
    r.held_red_joker[s] = r.hands[s].count(kRJ) > 0;

  auto start_play = [&](int leader) {
    r.first_leader = static_cast<int8_t>(leader);
    r.to_move = static_cast<int8_t>(leader);
    r.phase = Phase::Play;
  };

  if (double_win) {
    const int reds = r.hands[third].count(kRJ) + r.hands[dweller].count(kRJ);
    if (reds >= 2) { r.anti_tribute = true; start_play(banker); return; }
    r.tribute_payers = {static_cast<int8_t>(third), static_cast<int8_t>(dweller)};
    r.tribute_receivers = {-1, -1};
  } else {
    if (r.hands[dweller].count(kRJ) >= 2) {
      r.anti_tribute = true; start_play(banker); return;
    }
    if (!rules_.tribute_between_partners && partner_of(dweller) == banker) {
      start_play(dweller); return;
    }
    r.tribute_payers = {static_cast<int8_t>(dweller), -1};
    r.tribute_receivers = {static_cast<int8_t>(banker), -1};
  }
  r.tribute_step = 0;
  r.phase = Phase::Tribute;
  r.to_move = r.tribute_payers[0];
}

bool Engine::needs_decision(const MatchState& m) const {
  const Phase p = m.round.phase;
  return p == Phase::Tribute || p == Phase::BackTribute || p == Phase::Play;
}

void Engine::legal_actions(const MatchState& m, std::vector<Action>& out) const {
  const RoundState& r = m.round;
  switch (r.phase) {
    case Phase::Tribute:
      generate_tribute(r.hands[r.to_move], r.level, rules_, out);
      return;
    case Phase::BackTribute:
      generate_back_tribute(r.hands[r.to_move], r.level, rules_, out);
      return;
    case Phase::Play:
      generate_moves(r.hands[r.to_move], r.level, r.top, actions_, rules_, out);
      return;
    default:
      return;
  }
}

// After a play, decide whether the trick is over and who leads next.
void Engine::close_trick_if_done(MatchState& m) const {
  RoundState& r = m.round;
  int others = 0;
  for (int s = 0; s < 4; ++s)
    if (r.active(s) && s != r.holder) ++others;
  if (r.passes < others) return;

  int leader;
  if (r.holder >= 0 && r.active(r.holder)) {
    leader = r.holder;
  } else if (r.holder >= 0 && r.active(partner_of(r.holder))) {
    leader = partner_of(r.holder);                 // teammate lead
  } else {
    leader = r.holder >= 0 ? r.holder : r.to_move;
    for (int i = 1; i <= 4; ++i) {
      const int s = (leader + i) % 4;
      if (r.active(s)) { leader = s; break; }
    }
  }
  r.top = Action{};
  r.holder = -1;
  r.passes = 0;
  r.to_move = static_cast<int8_t>(leader);
}

void Engine::advance_seat(MatchState& m) const {
  RoundState& r = m.round;
  for (int i = 1; i <= 4; ++i) {
    const int s = (r.to_move + i) % 4;
    if (r.active(s)) { r.to_move = static_cast<int8_t>(s); return; }
  }
}

bool Engine::round_over(const MatchState& m) const {
  const RoundState& r = m.round;
  if (r.num_finished >= 3) return true;
  if (r.num_finished >= 2) {
    // Both members of one team are out.
    for (int t = 0; t < 2; ++t) {
      int n = 0;
      for (int i = 0; i < r.num_finished; ++i) if (team_of(r.order[i]) == t) ++n;
      if (n == 2) return true;
    }
  }
  return false;
}

// Fills the tail of `order` with the seats that never ran out.
void Engine::seal_order(MatchState& m) const {
  RoundState& r = m.round;
  if (r.num_finished >= 4) return;
  if (rules_.double_win_tail == DoubleWinTail::AscendingSeat) {
    for (int s = 0; s < 4 && r.num_finished < 4; ++s)
      if (r.active(s)) {
        r.finish_pos[s] = r.num_finished;
        r.order[r.num_finished++] = static_cast<int8_t>(s);
      }
    return;
  }
  const int start = r.num_finished >= 2 ? (r.order[1] + 1) % 4 : 0;
  for (int i = 0; i < 4 && r.num_finished < 4; ++i) {
    const int s = (start + i) % 4;
    if (r.active(s)) {
      r.finish_pos[s] = r.num_finished;
      r.order[r.num_finished++] = static_cast<int8_t>(s);
    }
  }
}

void Engine::apply_play(MatchState& m, const Action& a) const {
  RoundState& r = m.round;
  const int seat = r.to_move;
  r.last_action[seat] = a;
  r.has_acted[seat] = true;
  ++r.steps;

  if (a.is_pass()) {
    ++r.passes;
    advance_seat(m);
    close_trick_if_done(m);
    return;
  }

  r.hands[seat].remove_all(a.cards);
  r.played[seat].add_all(a.cards);
  r.top = a;
  r.holder = static_cast<int8_t>(seat);
  r.passes = 0;

  if (r.hands[seat].empty()) {
    r.finish_pos[seat] = r.num_finished;
    r.order[r.num_finished++] = static_cast<int8_t>(seat);
    ++r.num_out;
    if (round_over(m)) {
      seal_order(m);
      r.phase = Phase::RoundEnd;
      return;
    }
  }
  advance_seat(m);
  close_trick_if_done(m);
}

void Engine::apply_tribute(MatchState& m, const Action& a) const {
  RoundState& r = m.round;
  const int payer = r.to_move;
  const CardId card = static_cast<CardId>(std::countr_zero(a.cards.has1));
  r.tribute_cards[r.tribute_step] = static_cast<int16_t>(card);
  ++r.tribute_step;

  const bool more = r.tribute_step < 2 && r.tribute_payers[r.tribute_step] >= 0;
  if (more) { r.to_move = r.tribute_payers[r.tribute_step]; return; }

  // Every tribute card is chosen: route them and move the cards.
  const int banker = m.prev_order[0];
  const int follower = m.prev_order[1];
  const bool double_win = partner_of(banker) == follower;
  int leader;
  if (double_win) {
    const int a_seat = r.tribute_payers[0], b_seat = r.tribute_payers[1];
    const int pa = power(rank_of(static_cast<CardId>(r.tribute_cards[0])), r.level);
    const int pb = power(rank_of(static_cast<CardId>(r.tribute_cards[1])), r.level);
    const TributeRouting route = double_tribute(banker, a_seat, pa, b_seat, pb, rules_);
    leader = route.leader;
    if (rules_.tribute_tie == TributeTie::LastFinisher && pa == pb)
      leader = m.prev_order[3];
    for (int i = 0; i < 2; ++i) {
      const int seat = r.tribute_payers[i];
      r.tribute_receivers[i] = static_cast<int8_t>(
          seat == route.to_banker ? banker : follower);
    }
  } else {
    leader = r.tribute_payers[0];
  }
  (void)payer;

  for (int i = 0; i < 2; ++i) {
    if (r.tribute_payers[i] < 0) continue;
    const int from = r.tribute_payers[i];
    const int to = r.tribute_receivers[i];
    const CardId c = static_cast<CardId>(r.tribute_cards[i]);
    r.hands[from].remove(c);
    r.hands[to].add(c);
    r.tribute_moves[r.num_tribute_moves++] = TributeMove{
        static_cast<int8_t>(from), static_cast<int8_t>(to), c, false};
  }
  r.first_leader = static_cast<int8_t>(leader);
  r.tribute_step = 0;
  r.phase = Phase::BackTribute;
  r.to_move = r.tribute_receivers[0];
}

void Engine::apply_back_tribute(MatchState& m, const Action& a) const {
  RoundState& r = m.round;
  const int giver = r.to_move;
  const CardId card = static_cast<CardId>(std::countr_zero(a.cards.has1));
  const int receiver = r.tribute_payers[r.tribute_step];
  r.hands[giver].remove(card);
  r.hands[receiver].add(card);
  r.tribute_moves[r.num_tribute_moves++] = TributeMove{
      static_cast<int8_t>(giver), static_cast<int8_t>(receiver), card, true};
  ++r.tribute_step;

  if (r.tribute_step < 2 && r.tribute_payers[r.tribute_step] >= 0) {
    r.to_move = r.tribute_receivers[r.tribute_step];
    return;
  }
  r.phase = Phase::Play;
  r.to_move = r.first_leader;
}

// A seat whose only option is pass never gets asked (DESIGN.md 5.2).
void Engine::skip_forced(MatchState& m) const {
  if (!auto_pass_) return;
  // thread_local so that one Engine can drive many environments in parallel
  // without allocating in the rollout loop.
  static thread_local std::vector<Action> scratch;
  for (int guard = 0; guard < 64; ++guard) {
    if (m.round.phase != Phase::Play) return;
    scratch.clear();
    generate_moves(m.round.hands[m.round.to_move], m.round.level, m.round.top,
                   actions_, rules_, scratch);
    if (scratch.size() != 1 || !scratch[0].is_pass()) return;
    apply_play(m, scratch[0]);
  }
}

void Engine::apply(MatchState& m, const Action& a) const {
  switch (m.round.phase) {
    case Phase::Tribute: apply_tribute(m, a); break;
    case Phase::BackTribute: apply_back_tribute(m, a); break;
    case Phase::Play: apply_play(m, a); break;
    default: return;
  }
  skip_forced(m);
}

void Engine::end_round(MatchState& m, RoundResult& out) const {
  RoundState& r = m.round;
  seal_order(m);
  out = RoundResult{};
  out.round_index = m.round_index;
  out.order = r.order;
  out.num_out = r.num_out;
  out.round_level = r.level;
  const LevelGain g = level_gain(r.order);
  out.winning_team = static_cast<int8_t>(g.team);
  out.gain = static_cast<int8_t>(g.gain);

  const EndOfRound e = end_of_round(m.levels, m.fails, m.owner, r.level, r.order, rules_);
  const bool zeroed = m.owner >= 0 && r.level == 12 && m.levels[m.owner] == 12 &&
                      g.team == m.owner && e.match_winner < 0;
  m.levels = e.levels;
  m.fails = e.fails;
  m.owner = e.next_owner;
  m.winner = e.match_winner;
  out.levels = e.levels;
  out.fails = e.fails;
  out.next_owner = e.next_owner;
  out.match_winner = e.match_winner;

  // The DMC target (DESIGN.md 8.2). A level-A round the owner wins with Banker
  // and Dweller cannot pass A, so it scores nothing for either team.
  for (int s = 0; s < 4; ++s) {
    const int v = zeroed ? 0 : (team_of(s) == g.team ? g.gain : -g.gain);
    out.seat_return[s] = static_cast<int8_t>(v);
  }

  m.prev_order = r.order;
  m.has_prev = true;
  if (m.winner >= 0) r.phase = Phase::MatchEnd;
}

}  // namespace gd
