"""The PPO learn step's host trims are bitwise: one backward call over policy +
value, the encoder diagnostic norms taken from the actor clip's own per-tensor
norm pass, and the stats stacked without per-value conversions give the same
gradients, clipping, Adam steps and statistics as the previous step."""
import pytest
import torch

from train.history_ppo import HistoryPPOConfig, HistoryTrainer


class PreviousStep:
    """The step as it was: the encoder norms in their own foreach pass,
    ``clip_grad_norm_`` over the actor, one ``.detach().to(device, float32)`` per
    statistic before the stack."""

    def clip_actor_gradients(self, parameters, encoder_index):
        grads = [parameters[i].grad.detach().float() for i in encoder_index
                 if parameters[i].grad is not None]
        encoder = torch.stack(torch._foreach_norm(grads)) ** 2 if grads else None
        return torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip), encoder

    def stats_to_host(self, values, encoder):
        encoder = () if encoder is None else encoder.unbind()
        return torch.stack([value.detach().to(self.device, torch.float32) for value in
                            (values[0], values[1], *encoder, *values[2:])]).tolist()


class PreviousStepTrainer(PreviousStep, HistoryTrainer):
    def backward_minibatch(self, rows, batch=None):
        terms = self.minibatch_loss(rows)
        terms["policy_total"].backward()
        terms["value_total"].backward()
        return terms


def config(**overrides) -> HistoryPPOConfig:
    values = dict(width=32, layers=1, heads=4, num_envs=4, steps_per_update=90, seed=11,
                  updates=3, epochs=2, minibatch_matches=1, snapshot_updates=1,
                  response_mode="auxiliary", aux_heads="next,belief,outcome")
    values.update(overrides)
    return HistoryPPOConfig(**values)


@pytest.mark.parametrize("grad_clip", [10.0, 0.05])     # 0.05: clipping always active
def test_trimmed_step_matches_previous_step_bitwise(tmp_path, grad_clip):
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True)
    try:
        new = HistoryTrainer(config(grad_clip=grad_clip), tmp_path / "new")
        old = PreviousStepTrainer(config(grad_clip=grad_clip), tmp_path / "old")
        for _ in range(3):
            a, b = new.update(), old.update()
            assert a["minibatches"] == b["minibatches"] > 0
            for key in HistoryTrainer.STAT_KEYS:
                assert a[key] == b[key], key
        if grad_clip < 1:
            assert a["actor_grad_norm"] > grad_clip
        for x, y in zip([*new.actor.parameters(), *new.critic.parameters()],
                        [*old.actor.parameters(), *old.critic.parameters()]):
            assert torch.equal(x, y)
        for name in ("actor_optimizer", "critic_optimizer"):
            for x, y in zip(getattr(new, name).state.values(), getattr(old, name).state.values()):
                assert all(torch.equal(x[k], y[k]) for k in x)
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def test_one_backward_call_gives_the_two_calls_gradients(tmp_path):
    trainer = HistoryTrainer(config(), tmp_path / "t")
    trainer.collect()
    values = trainer.refresh_values()
    trainer.buffer.finalize(values, trainer.config.gamma, trainer.config.gae_lambda)
    params = [*trainer.actor.parameters(), *trainer.critic.parameters()]
    for rows in list(trainer.minibatches())[:3]:
        grads = []
        for combined in (True, False):
            for p in params:
                p.grad = None
            terms = trainer.minibatch_loss(rows)
            if combined:
                (terms["policy_total"] + terms["value_total"]).backward()
            else:
                terms["policy_total"].backward()
                terms["value_total"].backward()
            grads.append([None if p.grad is None else p.grad.clone() for p in params])
        for a, b in zip(*grads):
            assert (a is None) == (b is None) and (a is None or torch.equal(a, b))


def _groups(trainer):
    return [g for o in (trainer.actor_optimizer, trainer.critic_optimizer) for g in o.param_groups]


def _steps_on(trainer, fused: bool) -> bool:
    return all(state["step"].device == (p.device if fused else torch.device("cpu"))
               for o in (trainer.actor_optimizer, trainer.critic_optimizer)
               for p, state in o.state.items())


def test_adam_fused_is_opt_in_and_optimizer_state_loads_both_ways(tmp_path):
    assert HistoryPPOConfig().adam_fused is False
    first = HistoryTrainer(config(updates=5, epochs=1), tmp_path / "a")
    assert all(g["fused"] is None for g in _groups(first))
    first.update()
    path = first.save(tmp_path / "a.pt")
    saved = torch.load(path, map_location="cpu", weights_only=False)["optimizer"]
    # foreach checkpoint -> fused optimizer: same moments, fused groups, device steps.
    fused = HistoryTrainer(config(updates=5, epochs=1), tmp_path / "b", resume=path,
                           resume_overrides={"adam_fused": True})
    assert fused.config.adam_fused and all(g["fused"] is True for g in _groups(fused))
    assert _steps_on(fused, True)
    params = list(fused.actor.parameters())
    for index, state in saved["actor"]["state"].items():
        loaded = fused.actor_optimizer.state[params[index]]
        assert all(torch.equal(state[k].float(), loaded[k].float()) for k in state)
    assert fused.config_changes[-1]["changes"]["adam_fused"] == [False, True]
    fused.update()
    path = fused.save(tmp_path / "b.pt")
    assert torch.load(path, map_location="cpu", weights_only=False)[
        "optimizer"]["actor"]["param_groups"][0]["fused"] is True
    # fused checkpoint -> foreach optimizer, and without an override it stays fused.
    back = HistoryTrainer(config(updates=5, epochs=1), tmp_path / "c", resume=path,
                          resume_overrides={"adam_fused": False})
    assert all(g["fused"] is None for g in _groups(back)) and _steps_on(back, False)
    back.update()
    kept = HistoryTrainer(config(updates=5, epochs=1), tmp_path / "d", resume=path)
    assert kept.config.adam_fused and all(g["fused"] is True for g in _groups(kept))


def test_adam_fused_changes_only_fp32_order(tmp_path):
    """Tier 2: from the same start and data, fused Adam's parameters agree with the
    default's up to FP32 noise (Adam lets noise on near-zero gradient elements
    move a few elements by up to about lr per step)."""
    a = HistoryTrainer(config(updates=1, epochs=1), tmp_path / "a")
    b = HistoryTrainer(config(updates=1, epochs=1, adam_fused=True), tmp_path / "b")
    la, lb = a.update(), b.update()
    assert la["update_samples"] == lb["update_samples"]
    assert abs(la["policy_loss"] - lb["policy_loss"]) <= 1e-6
    diff = torch.cat([(x - y).detach().abs().reshape(-1) for x, y in
                      zip([*a.actor.parameters(), *a.critic.parameters()],
                          [*b.actor.parameters(), *b.critic.parameters()])])
    assert diff.max() > 0                                   # really another kernel
    assert float((diff <= 1e-6).float().mean()) > 0.999
    assert float(diff.max()) <= 2 * a.config.lr * la["minibatches"]
