"""Experimental one-class artwork model. Never a production OK/NG model."""
from pathlib import Path
import hashlib,json,platform
import numpy as np
from PIL import Image,ImageOps
from scipy import ndimage
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
import sklearn,joblib
from . import core

VERSION='reference-patches-v1'

def patches(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('L');im.thumbnail((1024,1024));a=np.asarray(im,dtype=float)/255
    return patch_features(a)

def patch_features(a):
    h,w=a.shape;features=[]
    # Fixed grid retains every region, including background; no fitted preprocessing.
    for y in range(4):
        for x in range(4):
            p=a[y*h//4:(y+1)*h//4,x*w//4:(x+1)*w//4]
            gx=ndimage.sobel(p,axis=1);gy=ndimage.sobel(p,axis=0)
            hist=np.histogram(p,bins=16,range=(0,1))[0]/p.size
            features.append([float(p.mean()),float(p.std()),float((np.hypot(gx,gy)>.3).mean()),float(ndimage.laplace(p).var()),*hist.tolist()])
    return np.array(features)

def partition(rows):
    # Shared group or identical raster must not cross partitions.
    parent={r['group']:r['group'] for r in rows};seen={}
    def find(x):
        while parent[x]!=x:x=parent[x]
        return x
    for r in rows:
        h=r['image_sha256']
        if h in seen:parent[find(r['group'])]=find(seen[h])
        seen[h]=r['group']
    groups=[find(r['group']) for r in rows];idx=np.arange(len(rows))
    rest,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(idx,groups=groups))
    tr,cal=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=43).split(rest,groups=np.array(groups)[rest]))
    return {'train':rest[tr].tolist(),'calibration':rest[cal].tolist(),'test':test.tolist()}

def train(manifest_path,output):
    source=Path(manifest_path);root=source.parent;rows=json.loads(source.read_text())['items']
    for r in rows:r['image_sha256']=hashlib.sha256((root/r['image']).read_bytes()).hexdigest()
    splits=partition(rows);features=[patches(root/r['image']) for r in rows]
    model=IsolationForest(n_estimators=300,max_samples='auto',contamination='auto',random_state=42,n_jobs=1)
    model.fit(np.concatenate([features[i] for i in splits['train']]))
    # Image score = mean of four most anomalous grid regions. Higher means unusual.
    def score(f):return float(np.sort(-model.score_samples(f))[-4:].mean())
    scores=np.array([score(f) for f in features])
    threshold=float(np.quantile(scores[splits['calibration']],.95,method='higher'))
    test_scores=scores[splits['test']];alarms=int((test_scores>threshold).sum())
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    bundle={'model':model,'threshold':threshold,'feature_version':VERSION,'grid':4,'score':'mean_top4_negative_score_samples','domain':'pdf_artwork','production_ready':False}
    joblib.dump(bundle,out/'model.joblib')
    assignments=[]
    for name,indices in splits.items():
        for i in indices:
            assignments.append({**rows[i],'split':name,'anomaly_score':float(scores[i]),'flagged':bool(scores[i]>threshold)})
    paired=core.all_samples();ref_ids={r['source_sha256'] for r in rows}
    matched=sum(any(ref['id'] in ref_ids for ref in s.get('references',[])) for s in paired)
    report={'kind':'정상 PDF 시안 전용 이상탐지 실험','algorithm':'IsolationForest','feature_version':VERSION,'created_at':core.now(),
      'source_count':len(rows),'split_counts':{k:len(v) for k,v in splits.items()},'train_patch_count':16*len(splits['train']),
      'threshold':threshold,'threshold_rule':'calibration image score 95th percentile, higher; alarm when score > threshold',
      'test':{'normal_reference_count':len(test_scores),'flagged_count':alarms,'reference_alarm_rate':alarms/len(test_scores),'accuracy':None,'precision':None,'recall':None,'f1':None},
      'defect_photo_pair_candidates':matched,'defect_photos_evaluated':0,'production_ready':False,
      'python':platform.python_version(),'sklearn':sklearn.__version__,'seed':42,
      'manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'limitations':['정상 기준은 사용자가 지정한 PDF 시안이며 물리적 정상 촬영본이 아니다.','재단선·분판·치수 및 도메인 차이를 학습할 수 있다.','정합·OCR·결함 유형/위치 검출 모델이 아니다.','실제 불량 검수와 정상 실촬영 시험이 없어 Accuracy/Recall/F1은 산출하지 않았다.','목형 칼날에 사용할 수 없다.']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'dataset.json').write_text(json.dumps(assignments,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='output/reference-images/manifest.json');p.add_argument('--output',default='output/reference-model');a=p.parse_args()
    print(json.dumps(train(a.manifest,a.output),ensure_ascii=False,indent=2))
