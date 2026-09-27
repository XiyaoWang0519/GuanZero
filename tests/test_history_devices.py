"""Split rollout/learner placement preserves policy probabilities and resume."""
import numpy as np
import pytest
import torch

from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from train.history_population import weights_digest


def test_shared_cpu_device_uses_one_actor(tmp_path):
    cfg=HistoryPPOConfig(width=16,layers=1,heads=2,num_envs=2,steps_per_update=90,
                         epochs=1,torch_threads=1,rollout_device='cpu')
    trainer=HistoryTrainer(cfg,tmp_path,device='cpu')
    assert trainer.rollout_actor is trainer.actor
    assert trainer.collector.device.type=='cpu'
    assert trainer.payload()['rng']['sampler_device']=='cpu'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='split placement requires CUDA')
@pytest.mark.parametrize('mode',['none','auxiliary','explicit'])
def test_cpu_rollout_cuda_learning_sync_population_and_resume(tmp_path,mode):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=HistoryPPOConfig(width=16,layers=1,heads=2,num_envs=2,num_threads=1,
        steps_per_update=90,epochs=1,minibatch_matches=1,torch_threads=1,seed=29,
        causal_sdpa=True,rollout_kv_cache=True,snapshot_updates=1,
        response_mode=mode,rollout_device='cpu')
    trainer=HistoryTrainer(cfg,tmp_path/'start',device='cuda')
    assert next(trainer.actor.parameters()).is_cuda
    assert next(trainer.critic.parameters()).is_cuda
    assert next(trainer.rollout_actor.parameters()).device.type=='cpu'
    before={n:p.clone() for n,p in trainer.actor.state_dict().items()}
    trainer.update()
    assert any(not torch.equal(before[n],p) for n,p in trainer.actor.state_dict().items())
    # Force both historical identities through the real mirror path.
    for identity,model in trainer.population.models.items():
        mirror=trainer.resolve_rollout_policy(identity)
        assert weights_digest(mirror.state_dict())==weights_digest(model.state_dict())
        assert next(mirror.parameters()).device.type=='cpu'
    for _ in range(2):
        trainer.collect()
        assert weights_digest(trainer.rollout_actor.state_dict())==weights_digest(trainer.actor.state_dict())
        data=trainer.buffer.compact()
        rows=np.flatnonzero(data['version']==trainer.progress['updates'])[-50:]
        with torch.no_grad(): logp,_=trainer.recompute_log_probs(rows)
        np.testing.assert_allclose(logp.cpu().numpy(),data['logp'][rows],rtol=0,atol=3e-5)
        # A manual learn without advancing the counter must still refresh mirrors.
        stats=trainer.learn()
        assert stats['encoder_grad_norm']>0
        assert not trainer.collector.caches[0].entries
    saved=trainer.save()
    restored=HistoryTrainer(cfg,tmp_path/'resume',device='cuda',resume=saved)
    assert restored.rollout_device.type=='cpu'
    assert restored.payload()['rng']['sampler_device']=='cpu'
    assert torch.equal(restored.generator.get_state(),trainer.generator.get_state())
    assert weights_digest(restored.actor.state_dict())==weights_digest(trainer.actor.state_dict())
    assert not restored.collector.caches
    assert restored.update()['encoder_grad_norm']>0
