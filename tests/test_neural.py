import numpy as np,pytest,joblib
from api.neural import Autoencoder,NAMES,image_patches
@pytest.mark.parametrize('name',NAMES)
def test_gradient_finite_difference(name):
    model=Autoencoder(name);x=np.random.default_rng(7).random((2,16,16));loss,g=model.gradients(x)
    for k,p in model.p.items():
        index=tuple(0 for _ in p.shape);original=p[index];eps=1e-5
        p[index]=original+eps;plus=model.gradients(x)[0];p[index]=original-eps;minus=model.gradients(x)[0];p[index]=original
        assert g[k][index]==pytest.approx((plus-minus)/(2*eps),rel=1e-3,abs=1e-7),k
@pytest.mark.parametrize('name',NAMES)
def test_train_save_reload(name,tmp_path):
    x=np.ones((16,16,16))*.8;m=Autoencoder(name);before=m.gradients(x)[0];m.fit(x,x,epochs=5)
    assert m.gradients(x)[0]<before
    joblib.dump(m,tmp_path/'m');assert joblib.load(tmp_path/'m').score(x)==pytest.approx(m.score(x))
def test_patch_shape():assert image_patches(np.zeros((80,120))).shape==(16,16,16)
