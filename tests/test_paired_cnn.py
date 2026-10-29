import numpy as np,pytest
from scipy import ndimage
from api.paired_cnn import align,pair_tensor,PairedCNN

def test_alignment_translation_and_rejection():
    a=np.random.default_rng(2).random((64,64));q=ndimage.shift(a,(2,-1),order=0,cval=1)
    b,r=align(a,q,4);assert r['accepted'];assert r['shift']==[-2,1];assert np.mean((a-b)**2)<1e-12
    assert not align(a,q[:32])[1]['accepted']
    assert not align(a,np.random.default_rng(3).random(a.shape))[1]['accepted']

def test_paired_cnn_gradient():
    m=PairedCNN();x=np.random.default_rng(9).random((2,16,16,3));y=np.array([0,1]);_,g=m.gradients(x,y)
    for key,p in m.p.items():
        idx=tuple(0 for _ in p.shape);v=p[idx];eps=1e-5;p[idx]=v+eps;a=m.gradients(x,y)[0];p[idx]=v-eps;b=m.gradients(x,y)[0];p[idx]=v
        assert g[key][idx]==pytest.approx((a-b)/(2*eps),abs=1e-6,rel=1e-3)

@pytest.mark.parametrize('channels,kernel',[(12,3),(24,3),(6,5)])
def test_configurable_cnn_gradient(channels,kernel):
    m=PairedCNN(channels=channels,kernel=kernel);x=np.random.default_rng(9).random((2,12,12,3));y=np.array([0,1]);_,g=m.gradients(x,y)
    for key,p in m.p.items():
        idx=tuple(0 for _ in p.shape);v=p[idx];eps=1e-5;p[idx]=v+eps;a=m.gradients(x,y)[0];p[idx]=v-eps;b=m.gradients(x,y)[0];p[idx]=v
        assert g[key][idx]==pytest.approx((a-b)/(2*eps),abs=1e-6,rel=1e-3)

def test_sweep_changes_one_factor_and_threshold():
    from api.cnn_sweep import CONFIGS,BASE,choose_threshold
    for _,cfg in CONFIGS[1:]:assert sum(cfg[k]!=BASE[k] for k in BASE)==1
    y=np.array([0,0,1,1]);s=np.array([.1,.2,.5,.9]);t=choose_threshold(y,s)
    assert t==.2 and not (s[y==0]>t).any()
