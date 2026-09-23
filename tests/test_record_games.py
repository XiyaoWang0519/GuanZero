"""The game recorder replays cleanly and is reproducible."""
from eval.policies import load_policy
from eval.record_games import record_match, validate_round


def test_greedy_match_replays_and_is_deterministic():
    greedy = load_policy("greedy")
    games = [record_match("t", 7, ["a", "b"], [greedy, greedy], max_rounds=5) for _ in range(2)]
    assert games[0] == games[1]
    for rnd in games[0]["rounds"]:
        validate_round(rnd)
        assert len(rnd["hands"]) == 4 and all(len(h) == 27 for h in rnd["hands"])
