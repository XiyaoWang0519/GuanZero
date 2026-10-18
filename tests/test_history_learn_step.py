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
