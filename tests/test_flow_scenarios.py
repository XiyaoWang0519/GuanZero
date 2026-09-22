"""M0 task 6: the T-FLOW scenarios of RULES.md section 12.

Each scenario is scripted with an explicit DealSpec and short hands. The engine
does not require 27-card hands, which is what makes these readable.

All of them are played at level A (rank index 12), so the wild card is the
heart ace and the scripted hands simply avoid aces unless the case needs them.
"""
import gd
import gd_reference as g

LEVEL_A = 12


def _c(text):
    return g.cards(text)


def _state(hands, level=LEVEL_A, leader=0, prev_order=None,
           team_levels=(LEVEL_A, 5), fails=(0, 0), owner=0, rules=None):
    d = gd.DealSpec()
    d.hands = [sorted(h) for h in hands]
    d.level = level
    d.team_levels = list(team_levels)
    d.fails = list(fails)
    d.owner = owner
    if prev_order is not None:
        d.prev_order = list(prev_order)
    else:
        d.leader = leader
    m = gd.MatchState()
    e = gd.Engine(rules) if rules is not None else gd.Engine()
    e.set_deal(m, d)
    return e, m


def _pass(e, m):
    a = next(a for a in e.legal_actions(m) if a.is_pass)
    e.apply(m, a)


def _play(e, m, text):
    want = tuple(sorted(_c(text)))
    a = next(a for a in e.legal_actions(m) if tuple(sorted(a.cards)) == want)
    e.apply(m, a)
    return a


# ---- tricks and the teammate lead -----------------------------------------

def test_t_flow_01_holder_leads_the_next_trick():
    e, m = _state([_c("S3 D3 S7"), _c("S4 D4 C7"), _c("S6 D6 D7"), _c("SK DK H7")])
    _play(e, m, "S3 D3")          # seat 0 leads a pair
    _pass(e, m)                   # seat 1 could beat it and does not
    _pass(e, m)                   # seat 2 likewise
    _play(e, m, "SK DK")          # seat 3 takes the trick
    # Seats 0, 1 and 2 hold only singles now, so the engine passes for them.
    assert m.to_move == 3
    assert m.top_is_open


def test_t_flow_02_teammate_lead_after_going_out():
    e, m = _state([_c("S7"), _c("S8 D8"), _c("S9 D9"), _c("ST DT")])
    _play(e, m, "S7")             # seat 0 plays its last card
    assert m.order[0] == 0
    _pass(e, m)
    _pass(e, m)
    _pass(e, m)
    assert m.to_move == 2, "the partner of the seat that went out inherits the lead"
    assert m.top_is_open


def test_t_flow_03_overtaking_the_finishing_play():
    e, m = _state([_c("S7"), _c("S8 D8"), _c("S9 D9"), _c("ST DT")])
    _play(e, m, "S7")
    _play(e, m, "S8")             # seat 1 beats the last card
    _pass(e, m)
    _pass(e, m)
    assert m.to_move == 1
    assert m.top_is_open


# ---- round end -------------------------------------------------------------

def test_t_flow_04_double_win_ends_the_round_at_once():
    e, m = _state([_c("S7"), _c("S8 D8"), _c("S9"), _c("ST DT")])
    _play(e, m, "S7")             # seat 0 out
    _pass(e, m)                   # seat 1 declines
    _play(e, m, "S9")             # seat 2 out: both of team 0 are finished
    assert m.phase == gd.Phase.RoundEnd
    assert len(m.hand(1)) == 2 and len(m.hand(3)) == 2, "losers keep their cards"
    r = e.end_round(m)
    assert r.order[:2] == [0, 2]
    assert r.num_finished_seats == 2
    assert r.winning_team == 0 and r.gain == 3
    assert sum(r.seat_return) == 0


def test_t_flow_05_three_out_ends_the_round_with_a_dweller():
    e, m = _state([_c("S7"), _c("S8"), _c("S9 D9"), _c("ST")])
    _play(e, m, "S7")             # seat 0 out
    _play(e, m, "S8")             # seat 1 out
    _pass(e, m)                   # seat 2 declines
    _play(e, m, "ST")             # seat 3 out
    assert m.phase == gd.Phase.RoundEnd
    r = e.end_round(m)
    assert r.order[:3] == [0, 1, 3]
    assert r.order[3] == 2, "seat 2 is the Dweller"
    assert r.winning_team == 0 and r.gain == 1


# ---- tribute ---------------------------------------------------------------

def test_t_flow_06_anti_tribute_single():
    e, m = _state([_c("S3 D3"), _c("S4 D4"), _c("S6 D6"), _c("HR HR S9")],
                  prev_order=[0, 1, 2, 3])
    assert m.phase == gd.Phase.Play, "both red jokers with the Dweller: anti-tribute"
    assert m.to_move == 0, "the Banker leads"


def test_t_flow_07_anti_tribute_double_split_one_and_one():
    e, m = _state([_c("S3 D3"), _c("HR S4"), _c("S6 D6"), _c("HR S9")],
                  prev_order=[0, 2, 1, 3])
    assert m.phase == gd.Phase.Play
    assert m.to_move == 0


def test_t_flow_08_double_tribute_routes_by_card_power():
    e, m = _state([_c("S3 D3 S4"), _c("HR S6 D6"), _c("S7 D7 S8"), _c("SB S9 D9")],
                  prev_order=[0, 2, 1, 3])
    assert m.phase == gd.Phase.Tribute
    assert m.to_move == 1
    _play(e, m, "HR")             # the Third pays
    assert m.to_move == 3
    _play(e, m, "SB")             # the Dweller pays
    # Both cards have moved and both receivers now owe a card back.
    assert m.phase == gd.Phase.BackTribute
    assert g.cid("HR") in m.hand(0), "the higher card goes to the Banker"
    assert g.cid("SB") in m.hand(2), "the other goes to the Follower"
    assert m.to_move == 0
    back0 = _play(e, m, "S3")
    assert m.to_move == 2
    _play(e, m, "S7")
    assert m.phase == gd.Phase.Play
    assert g.cid("S3") in m.hand(1), "the Banker returns to its own payer"
    assert g.cid("S7") in m.hand(3)
    assert m.to_move == 1, "the payer of the higher card leads"
    assert back0 is not None


_FLOW_10_HANDS = [_c("S3 D3 S4"), _c("SB S6 D6"), _c("S7 D7 S8"), _c("SB S9 D9")]


def _flow_10(rules=None):
    """Seats 1 and 3 both pay SB after a double win by 0 and 2. The Banker
    returns S3 and the Follower returns S7, so where those land shows who paid
    whom. Returns the state once play starts."""
    e, m = _state(_FLOW_10_HANDS, prev_order=[0, 2, 1, 3], rules=rules)
    assert m.phase == gd.Phase.Tribute
    _play(e, m, "SB")             # seat 1
    _play(e, m, "SB")             # seat 3
    assert m.hand(0).count(g.cid("SB")) == 1
    assert m.hand(2).count(g.cid("SB")) == 1
    # Receivers return in payer order, so who moves first depends on pairing.
    for _ in range(2):
        assert m.phase == gd.Phase.BackTribute
        _play(e, m, "S3" if m.to_move == 0 else "S7")
    assert m.phase == gd.Phase.Play
    return m


def test_house_default_tribute_tie_is_downstream():
    assert gd.RuleConfig.house().tribute_tie == gd.TributeTie.Downstream
    assert gd.RuleConfig().tribute_tie == gd.TributeTie.Downstream
    assert gd.RuleConfig.ogd().tribute_tie == gd.TributeTie.LastFinisher


def test_t_flow_10_double_tribute_tie_goes_clockwise():
    m = _flow_10()
    # Equal power, official rule: each loser pays its upstream neighbour, so
    # the Banker's downstream seat 1 pays the Banker and leads, 3 pays 2.
    assert g.cid("S3") in m.hand(1), "the Banker returns to seat 1, its payer"
    assert g.cid("S7") in m.hand(3), "the Follower returns to seat 3, its payer"
    assert m.to_move == 1, "the Banker's downstream seat leads on a tie"


def test_t_flow_10_under_tribute_tie_upstream():
    rules = gd.RuleConfig.house()
    rules.tribute_tie = gd.TributeTie.Upstream
    m = _flow_10(rules)
    assert g.cid("S3") in m.hand(3), "the Banker returns to seat 3, its payer"
    assert g.cid("S7") in m.hand(1)
    assert m.to_move == 3, "the upstream seat leads on a tie"


def test_t_flow_10_under_the_ogd_profile_leads_from_the_last_finisher():
    rules = gd.RuleConfig.ogd()
    e, m = _state([_c("S3 D3 S4"), _c("SB S6 D6"), _c("S7 D7 S8"), _c("SB S9 D9")],
                  prev_order=[0, 2, 1, 3], rules=rules)
    _play(e, m, "SB")
    _play(e, m, "SB")
    while m.phase == gd.Phase.BackTribute:
        e.apply(m, e.legal_actions(m)[0])
    assert m.to_move == 3, "prev_order[3] is seat 3 here, so both profiles agree"


# ---- match bookkeeping at level A -----------------------------------------

def test_t_flow_09_banker_and_dweller_at_a_is_a_failed_attempt():
    levels, fails, owner, winner = gd.end_of_round([12, 5], [0, 0], 0, 12, [0, 1, 3, 2])
    assert winner is None
    assert levels == [12, 5]
    assert fails == [1, 0]
    assert owner == 0, "the owner keeps the round"


def test_t_flow_11_a_round_at_a_lost_by_the_owner():
    levels, fails, owner, winner = gd.end_of_round([12, 12], [0, 0], 1, 12, [0, 2, 1, 3])
    assert winner is None
    assert fails == [0, 1]
    assert owner == 0


def test_house_default_has_no_a_fail_reset():
    assert gd.RuleConfig.house().a_fail_limit == 0
    assert gd.RuleConfig.ogd().a_fail_limit == 4


def test_t_flow_12_third_failure_does_not_reset_under_house_rules():
    levels, fails, owner, winner = gd.end_of_round([12, 5], [2, 0], 0, 12, [1, 3, 0, 2])
    assert winner is None
    assert levels == [12, 8], "the owner stays at A, the winners promote"
    assert fails == [3, 0]
    assert owner == 1


def test_t_flow_12_third_failure_resets_the_owner_with_limit_3():
    rules = gd.RuleConfig.house()
    rules.a_fail_limit = 3
    levels, fails, owner, winner = gd.end_of_round([12, 5], [2, 0], 0, 12, [1, 3, 0, 2],
                                                   rules)
    assert winner is None
    assert levels == [0, 8], "the owner drops to the deuce, the winners promote"
    assert fails == [0, 0]
    assert owner == 1


def test_t_flow_13_winning_a_round_at_the_other_level_is_not_an_attempt():
    levels, fails, owner, winner = gd.end_of_round([12, 11], [1, 0], 1, 11, [0, 2, 1, 3])
    assert winner is None
    assert levels == [12, 11]
    assert fails == [1, 0], "no failure: the round was not the owner's attempt"
    assert owner == 0


def test_passing_a_requires_winning_your_own_round():
    assert gd.end_of_round([12, 5], [0, 0], 0, 12, [0, 1, 2, 3])[3] == 0
    assert gd.end_of_round([12, 5], [0, 0], 0, 12, [0, 2, 1, 3])[3] == 0


def test_ogd_profile_does_not_count_a_lost_a_round():
    ogd = gd.RuleConfig.ogd()
    house = gd.RuleConfig.house()
    lost = ([12, 12], [0, 0], 1, 12, [0, 2, 1, 3])
    assert gd.end_of_round(*lost, rules=house)[1] == [0, 1]
    assert gd.end_of_round(*lost, rules=ogd)[1] == [0, 0]
