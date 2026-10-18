from types import SimpleNamespace
import numpy as np
import torch
from bench.qknorm_fusion_variants import fusion_states, calibration, VARIANTS


class Zero(torch.nn.Module):
    def forward(self,x):return torch.zeros_like(x)


def actor():
    return SimpleNamespace(attention_norm=torch.nn.LayerNorm(8),feed_forward=Zero(),output_norm=torch.nn.Identity())


def test_all_variants_share_the_calibration_anchor_and_preserve_width():
    torch.manual_seed(71);a=actor();ref=dict(query=torch.randn(8),reader=torch.randn(8))
    q=ref['query'].repeat(3,1);r=ref['reader'].repeat(3,1)
    result=fusion_states(a,q,r,ref)
    assert set(result)==set(VARIANTS)
    for state in result.values():
        assert state.shape==(3,8) and torch.isfinite(state).all()
        torch.testing.assert_close(state,result['native'],rtol=0,atol=1e-6)


def test_postnorm_residual_preserves_uniform_history_component_that_ln_removes():
    torch.manual_seed(72);a=actor();q=torch.randn(1,8);r=torch.randn(1,8)
    ref=dict(query=q[0].clone(),reader=r[0].clone())
    base=fusion_states(a,q,r,ref);changed=fusion_states(a,q,r+.2,ref)
    torch.testing.assert_close(changed['native'],base['native'],atol=1e-6,rtol=0)
    expected=.2/torch.sqrt((q+r).var(1,unbiased=False,keepdim=True)+a.attention_norm.eps)
    torch.testing.assert_close(changed['postnorm1']-base['postnorm1'],expected.expand_as(q),atol=1e-6,rtol=0)
    torch.testing.assert_close(changed['restore_mean']-base['restore_mean'],expected.expand_as(q),atol=1e-6,rtol=0)


def test_calibration_anchors_do_not_use_heldout_values():
    d=dict(wave=np.array([0,1,8,9]),context=np.zeros((4,260)),reader=np.zeros((4,128)))
    first=calibration(d)
    d['context'][2:]=10000;d['reader'][2:]=-9999
    second=calibration(d)
    for k in first:torch.testing.assert_close(first[k],second[k],rtol=0,atol=0)
