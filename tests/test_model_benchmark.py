import joblib
import numpy as np
import pytest
from api.model_benchmark import Detector,NAMES,perturb

@pytest.mark.parametrize('name',NAMES)
def test_models_save_and_score_unseen(name,tmp_path):
    rng=np.random.default_rng(42);x=rng.normal(size=(100,20));new=rng.normal(size=(16,20))
    d=Detector(name).fit(x);before=d.scaler.mean_.copy();score=d.score(new)
    assert np.isfinite(score);assert np.array_equal(before,d.scaler.mean_)
    joblib.dump(d,tmp_path/'model');loaded=joblib.load(tmp_path/'model')
    assert loaded.score(new)==pytest.approx(score)
    assert np.allclose(before,x.mean(axis=0))

def test_synthetic_perturbations_preserve_source():
    a=np.random.default_rng(5).random((128,128));before=a.copy()
    for kind in ['erase','spot','blur']:
        b=perturb(a,kind)
        assert b.shape==a.shape and not np.array_equal(a,b)
        assert np.array_equal(a,before)
        assert 0<=b.min()<=b.max()<=1
