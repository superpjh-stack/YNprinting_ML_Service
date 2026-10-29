import json
import joblib
import numpy as np
from PIL import Image
from api.reference_training import train,patches,partition

def test_reference_training_is_separate_and_leak_free(tmp_path,monkeypatch):
    from api import core
    monkeypatch.setattr(core,'ROOT',tmp_path/'isolated')
    rows=[]
    for i in range(20):
        name=f'{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0,255,(120,160),dtype=np.uint8)).save(tmp_path/name)
        rows.append(dict(id=str(i),group=f'g{i}',image=name,source_sha256=str(i)))
    p=tmp_path/'manifest.json';p.write_text(json.dumps({'items':rows}));out=tmp_path/'out'
    report=train(p,out);data=json.loads((out/'dataset.json').read_text())
    sets=[{r['group'] for r in data if r['split']==s} for s in ['train','calibration','test']]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert sum(report['split_counts'].values())==20
    assert report['test']['recall'] is None and not report['production_ready']
    bundle=joblib.load(out/'model.joblib')
    assert bundle['model'].score_samples(patches(tmp_path/'0.png')).shape==(16,)
    calibration=[r['anomaly_score'] for r in data if r['split']=='calibration']
    assert report['threshold']==float(np.quantile(calibration,.95,method='higher'))
    assert core.overview()['runs']==[]

def test_identical_references_share_split():
    rows=[{'group':f'g{i}','image_sha256':f'h{i//2}'} for i in range(20)]
    parts=partition(rows)
    hashes=[{rows[i]['image_sha256'] for i in indices} for indices in parts.values()]
    assert not hashes[0]&hashes[1] and not hashes[0]&hashes[2] and not hashes[1]&hashes[2]
