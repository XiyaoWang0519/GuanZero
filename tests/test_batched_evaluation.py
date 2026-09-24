"""Compare the independent scalar and VecEnv evaluation paths end to end."""
import gd
import numpy as np
import pytest
import torch

from eval.arena import play_matches
from eval.batched import BatchActor, EvalConfig, play_duplicate_batch, play_matches_batch
from eval.duplicate import generate_deals, play_duplicate_teams
from eval.policies import GreedyPolicy, ModelPolicy, PrunedPolicy, RandomPolicy, StyledPolicy, load_policy
from eval.tribute import generate_tribute_deals
from train.model import GuandanModel, ModelConfig
from train.policy import StageBPolicy, PolicyConfig


def model_policy(pruned=False, learned=False, sample=False, margin=0):
    torch.set_num_threads(1)
    torch.manual_seed(51)
    model = GuandanModel(ModelConfig(state_width=16, state_layers=1, action_width=16,
                                     action_layers=1, fusion_width=16, fusion_layers=1))
    if pruned:
        return PrunedPolicy(StageBPolicy(model, model, PolicyConfig(top_k=5)),
                            heuristic_tribute=not learned, sample=sample, margin=margin)
    return ModelPolicy(model, heuristic_tribute=not learned, margin=margin)


@pytest.mark.parametrize('pruned,learned', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('wave,threads', [(1, 1), (7, 2)])
def test_duplicate_all_fields_match_reference(pruned, learned, wave, threads):
    agent = model_policy(pruned, learned)
    team = (agent, load_policy('styled:high-lead'))
    opponents = (GreedyPolicy(), load_policy('styled:bomb-happy'))
    deals = generate_deals(9, 33) + generate_tribute_deals(7, 33)
    expected = [play_duplicate_teams(d, team, opponents, 101 + i) for i, d in enumerate(deals)]
    assert play_duplicate_batch(deals, team, opponents, 101, EvalConfig(batch_size=wave, engine_threads=threads)) == expected


@pytest.mark.parametrize('pruned,learned', [(False, False), (True, True)])
def test_full_match_seed_schedule_counters_and_order(pruned, learned):
    agent = model_policy(pruned, learned)
    opponent = load_policy('styled:low-lead')
    indices = [3, 0, 7]
    expected = play_matches(agent, opponent, indices, seed=81)
    assert play_matches_batch(agent, opponent, indices, seed=81, config=EvalConfig(batch_size=2)) == expected


@pytest.mark.parametrize('kind', ['random', 'sample', 'margin', 'pruned-margin', 'styled'])
def test_stochastic_policies_replay_with_fixed_batch_config(kind):
    if kind == 'random':
        policy = RandomPolicy()
    elif kind == 'styled':
        style = np.asarray(gd.StyleParams.neutral().to_array(), np.float32)
        style[gd.STYLE_TEMPERATURE] = 0.8
        policy = StyledPolicy(style)
    else:
        policy = model_policy(pruned=kind != 'margin', sample=kind == 'sample',
                              margin=0 if kind == 'sample' else 0.1)
    args = (generate_deals(13, 77), (policy, policy), (GreedyPolicy(), GreedyPolicy()))
    first = play_duplicate_batch(*args, seed=42, config=EvalConfig(batch_size=5))
    assert play_duplicate_batch(*args, seed=42, config=EvalConfig(batch_size=5)) == first
    assert len(first) == 13


def test_bounds_and_unsupported_policy_fail_explicitly():
    with pytest.raises(ValueError):
        EvalConfig(batch_size=0)
    with pytest.raises(ValueError):
        EvalConfig(engine_threads=0)
    class Custom(GreedyPolicy):
        pass
    with pytest.raises(TypeError, match='scalar'):
        BatchActor(Custom(), 0)
    greedy = GreedyPolicy()
    with pytest.raises(RuntimeError, match='exceeded 1 rounds'):
        play_matches_batch(greedy, greedy, range(2), max_rounds=1)
    with pytest.raises(RuntimeError, match='exceeded 1 decisions'):
        play_duplicate_batch(generate_deals(2), (greedy, greedy), (greedy, greedy), max_decisions=1)
    env = gd.VecEnv(2)
    with pytest.raises(ValueError):
        env.reset(match_seeds=[1])
    with pytest.raises(ValueError):
        env.reset(generate_deals(2), match_seeds=[1, 2])


def test_nonfinite_model_scores_rejected():
    policy = model_policy()
    with torch.no_grad():
        for p in policy.model.parameters():
            p.fill_(float('nan'))
    with pytest.raises(ValueError, match='finite'):
        play_duplicate_batch(generate_deals(1), (policy, policy), (GreedyPolicy(), GreedyPolicy()))


@pytest.mark.parametrize('pruned', [False, True])
def test_sampling_frequencies_agree_with_scalar_policy(pruned):
    """Independent scalar draws vs repeated copies of one live VecEnv row."""
    import random
    from types import SimpleNamespace
    policy = model_policy(pruned=pruned, sample=pruned, margin=0 if pruned else 100)
    engine, state = gd.Engine(), gd.MatchState()
    deal = generate_deals(1, 123)[0]
    engine.set_deal(state, deal)
    actions = engine.legal_actions(state)
    obs = np.asarray(state.observation(state.to_move), np.float32)
    cand = np.stack([engine.encode_action(a, state, state.to_move) for a in actions])
    n = 4096
    batch = SimpleNamespace(obs=np.tile(obs, (n, 1)), cand=np.tile(cand, (n, 1)),
                            offsets=np.arange(n + 1, dtype=np.int64) * len(actions),
                            phase=np.full(n, int(state.phase)), greedy_choice=np.zeros(n, np.int32))
    actual = BatchActor(policy, 42).act(batch, np.arange(n))
    rng = random.Random(91)
    expected = [policy.select(engine, state, actions, rng) for _ in range(n)]
    a = np.bincount(actual, minlength=len(actions)) / n
    b = np.bincount(expected, minlength=len(actions)) / n
    # Six standard errors per category plus a finite-sample floor. This checks
    # support and categorical frequencies without demanding identical draws.
    se = np.sqrt((a * (1 - a) + b * (1 - b)) / n)
    assert np.all(np.abs(a - b) < 6 * se + 0.005)
    if pruned:
        assert set(actual) == set(expected)


def test_tied_logits_and_pruning_preserve_first_candidate():
    policy = model_policy(pruned=True, learned=True)
    with torch.no_grad():
        for model in (policy.stage_b.net, policy.stage_b.reference):
            for parameter in model.parameters():
                parameter.zero_()
    team, opponents = (policy, policy), (GreedyPolicy(), GreedyPolicy())
    deals = generate_deals(5, 12) + generate_tribute_deals(5, 12)
    expected = [play_duplicate_teams(d, team, opponents) for d in deals]
    assert play_duplicate_batch(deals, team, opponents) == expected


def test_suite_backends_preserve_reports_and_scalar_stochastic_replay():
    from eval.stage_b_baseline import evaluate_pair
    from eval.crossplay import evaluate_lineup
    from eval.duplicate import evaluate_duplicates
    scalar, batched = EvalConfig(backend='scalar'), EvalConfig(batch_size=3)
    reports = [evaluate_pair('styled:high-lead', 'greedy', 9, 3, 13, 20, eval_config=c)
               for c in (scalar, batched)]
    for report in reports:
        report.pop('seconds')
        report.pop('evaluation')
    assert reports[0] == reports[1]
    lineup = ('styled:high-lead', 'styled:low-lead')
    reports = [evaluate_lineup(lineup, ('greedy', 'greedy'), 9, 13, 20, eval_config=c)
               for c in (scalar, batched)]
    for report in reports:
        report.pop('seconds')
        report.pop('evaluation')
    assert reports[0] == reports[1]
    expected = evaluate_duplicates(RandomPolicy(), GreedyPolicy(), generate_deals(9, 13), 13, 20)
    expected.pop('pair_scores')
    expected.pop('results')
    assert evaluate_pair('random', 'greedy', 9, 0, 13, 20, eval_config=scalar)['duplicate'] == expected
