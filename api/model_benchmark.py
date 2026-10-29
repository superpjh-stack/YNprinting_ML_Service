"""Five normal-only baselines; synthetic sensitivity is not field performance."""
from pathlib import Path
import json,hashlib,time
import joblib,numpy as np
from PIL import Image,ImageOps
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score,average_precision_score
from .reference_training import partition,patch_features,VERSION
from .training import metrics
from . import core

class Detector:
    def __init__(self,name):self.name=name
    def fit(self,x):
        self.scaler=StandardScaler().fit(x);z=self.scaler.transform(x)
        models={'IsolationForest':IsolationForest(n_estimators=300,random_state=42,n_jobs=1),
         'OneClassSVM':OneClassSVM(kernel='rbf',nu=.05,gamma='scale'),
         'LOF':LocalOutlierFactor(n_neighbors=20,novelty=True,n_jobs=1),
         'kNN':NearestNeighbors(n_neighbors=5,n_jobs=1),
         'PCA':PCA(n_components=.95,svd_solver='full')}
        self.model=models[self.name].fit(z);return self
    def score(self,x):
        z=self.scaler.transform(x)
        if self.name=='kNN':s=self.model.kneighbors(z)[0].mean(axis=1)
        elif self.name=='PCA':s=np.square(z-self.model.inverse_transform(self.model.transform(z))).mean(axis=1)
        else:s=-self.model.score_samples(z)
        return float(np.sort(s)[-4:].mean())

NAMES=['IsolationForest','OneClassSVM','LOF','kNN','PCA']
KINDS=['erase','spot','blur']

def read_array(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('L');im.thumbnail((1024,1024));return np.asarray(im,dtype=float)/255

def perturb(a,kind):
    # Fixed, reproducible image perturbations. No hand-labelled real defects.
    out=a.copy();h,w=a.shape;sy=max(3,int(h*.06));sx=max(3,int(w*.06))
    energy=ndimage.uniform_filter(np.square(ndimage.sobel(a,axis=0))+np.square(ndimage.sobel(a,axis=1)),size=max(3,min(sx,sy)))
    energy[:sy]=0;energy[-sy:]=0;energy[:,:sx]=0;energy[:,-sx:]=0
    y,x=np.unravel_index(energy.argmax(),energy.shape);y=min(max(y-sy//2,0),h-sy);x=min(max(x-sx//2,0),w-sx)
    if kind=='erase':out[y:y+sy,x:x+sx]=1.
    elif kind=='spot':out[y:y+sy,x:x+sx]=0.
    elif kind=='blur':out=ndimage.gaussian_filter(a,sigma=2.)
    else:raise ValueError(kind)
    return out

def evaluate(det,features,synthetics,indices,threshold):
    ns=[det.score(features[i]) for i in indices];bytype={k:[det.score(synthetics[i][k]) for i in indices] for k in KINDS}
    scores=ns+sum(bytype.values(),[]);y=[0]*len(ns)+[1]*(len(ns)*len(KINDS));pred=[int(s>threshold) for s in scores]
    return {'normal_n':len(ns),'normal_alarm_count':sum(s>threshold for s in ns),'normal_alarm_rate':sum(s>threshold for s in ns)/len(ns),
      'synthetic':{**metrics(y,pred),'roc_auc':float(roc_auc_score(y,scores)),'average_precision':float(average_precision_score(y,scores)),
      'per_type_recall':{k:sum(s>threshold for s in v)/len(v) for k,v in bytype.items()}},
      'scores':{'normal':ns,**bytype}}

def run(manifest='output/reference-images/manifest.json',output='output/model-benchmark'):
    p=Path(manifest);rows=json.loads(p.read_text())['items'];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for r in rows:r['image_sha256']=hashlib.sha256((p.parent/r['image']).read_bytes()).hexdigest()
    parts=partition(rows);features=[];synthetics=[];extraction=[]
    for i,r in enumerate(rows):
        t=time.perf_counter();a=read_array(p.parent/r['image']);features.append(patch_features(a));extraction.append((time.perf_counter()-t)*1000)
        variants={k:perturb(a,k) for k in KINDS} if i not in parts['train'] else {}
        synthetics.append({k:patch_features(v) for k,v in variants.items()})
        if i in parts['test'][:2]:
            for k,v in variants.items():Image.fromarray((v*255).astype('uint8')).save(out/f'example_{i}_{k}.png')
    train_x=np.concatenate([features[i] for i in parts['train']]);results=[]
    for name in NAMES:
        t=time.perf_counter();det=Detector(name).fit(train_x);fit_s=time.perf_counter()-t
        calibration=[det.score(features[i]) for i in parts['calibration']];threshold=float(np.quantile(calibration,.95,method='higher'))
        val=evaluate(det,features,synthetics,parts['calibration'],threshold)
        test=evaluate(det,features,synthetics,parts['test'],threshold)
        timings=[]
        for i in parts['test']:
            for _ in range(3):
                t=time.perf_counter();det.score(features[i]);timings.append((time.perf_counter()-t)*1000)
        joblib.dump({'detector':det,'threshold':threshold,'feature_version':VERSION,'production_ready':False},out/f'{name}.joblib')
        results.append({'name':name,'threshold':threshold,'fit_seconds':fit_s,'inference_ms_p50':float(np.median(timings)),'inference_ms_p95':float(np.quantile(timings,.95)),
          'validation':val,'test':test})
    # Declare candidate using calibration synthetic F1 only, not test results.
    ranking=sorted(results,key=lambda r:(-(r['validation']['synthetic']['f1'] or 0),r['name']))
    report={'kind':'정상 시안 5개 모델 탐색 비교','created_at':core.now(),'source_count':len(rows),'split_counts':{k:len(v) for k,v in parts.items()},
      'feature_version':VERSION,'scaling':'StandardScaler fit on train patches only','selection_rule':'calibration synthetic F1 descending; tie alphabetical name',
      'recommended_experimental_model':ranking[0]['name'],'production_ready':False,'normal_test_previously_examined':True,
      'feature_extraction_ms_p50':float(np.median(extraction)),'feature_extraction_ms_p95':float(np.quantile(extraction,.95)),
      'synthetic_protocol':{'kinds':KINDS,'erase_spot':'6% width x 6% height at high edge energy; white/black rectangle','blur':'Gaussian sigma 2 at max dimension 1024px','per_split':{k:len(parts[k])*len(KINDS) for k in ['calibration','test']},'note':'Known perturbations, not human-confirmed defect labels; 25% normal / 75% synthetic mixture'},
      'models':results,'limitations':['현장 정상/불량 성능 아님','시험 시안 11장은 이전 실험에서 이미 관찰한 탐색 시험 자료','정상 시험 11장으로 오경보율 불확실성이 큼','합성 유형과 강도가 실제 결함 분포를 대표하지 않음','문안 의미·미세 문자·칼날 결함 탐지 검증 아님','같은 원본에서 파생한 정상/합성 샘플은 독립 표본이 아님'],
      'sources':['https://scikit-learn.org/stable/modules/outlier_detection.html','https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'dataset.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'items':rows,'splits':parts},ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    from api.model_benchmark import run as benchmark_run
    r=benchmark_run();print(json.dumps({'recommended':r['recommended_experimental_model'],'models':[{'name':m['name'],'fpr':m['test']['normal_alarm_rate'],'synthetic_f1':m['test']['synthetic']['f1'],'synthetic_recall':m['test']['synthetic']['recall'],'auc':m['test']['synthetic']['roc_auc'],'p95_ms':m['inference_ms_p95']} for m in r['models']]},indent=2))
