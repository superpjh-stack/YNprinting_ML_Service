from pathlib import Path
import json,hashlib,time
import numpy as np,joblib
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from .paired_cnn import PairedCNN,align,pair_tensor
from .reference_training import partition
from .training import metrics
from . import core

def rois(path):
    # Native 300dpi raster. No 16x16 downsampling.
    with Image.open(path) as im:a=np.asarray(im.convert('L'),dtype=float)/255
    tiles=[]
    for y in range(0,a.shape[0]-63,64):
        for x in range(0,a.shape[1]-63,64):
            p=a[y:y+64,x:x+64];tiles.append((float(p.std()),x,y,p.copy()))
    return sorted(tiles,key=lambda t:t[0],reverse=True)[:4]

def build(rows,root,parts):
    sets={k:{'x':[],'y':[],'meta':[]} for k in parts};failures=[]
    for split,ids in parts.items():
        for i in ids:
            for ri,(_,x,y,ref) in enumerate(rois(root/rows[i]['image'])):
                rng=np.random.default_rng(42+i*101+ri)
                for kind in ['normal','erase','spot','blur']:
                    q=ref.copy()
                    if kind=='erase':q[28:36,28:36]=1
                    elif kind=='spot':q[30:34,30:34]=0
                    elif kind=='blur':q=ndimage.gaussian_filter(q,1.2)
                    if kind!='normal' and np.max(np.abs(q-ref))<.02:failures.append({'id':rows[i]['id'],'kind':kind,'reason':'unchanged'});continue
                    shift=rng.integers(-2,3,size=2);q=np.clip(q*rng.uniform(.98,1.02)+rng.uniform(-.005,.005),0,1);q=ndimage.shift(q,shift,order=0,cval=1.)
                    aligned,quality=align(ref,q,4)
                    if not quality['accepted']:failures.append({'id':rows[i]['id'],'split':split,'kind':kind,**quality});continue
                    sets[split]['x'].append(pair_tensor(ref,aligned));sets[split]['y'].append(int(kind!='normal'));sets[split]['meta'].append({'id':rows[i]['id'],'group':rows[i]['group'],'roi':[x,y,64,64],'kind':kind,'alignment':quality})
    for s in sets.values():s['x']=np.array(s['x']);s['y']=np.array(s['y'])
    return sets,failures

def run():
    p=Path('output/reference-images/manifest.json');rows=json.loads(p.read_text())['items'];out=Path('output/paired-cnn');out.mkdir(parents=True,exist_ok=True)
    for r in rows:r['image_sha256']=hashlib.sha256((p.parent/r['image']).read_bytes()).hexdigest()
    parts=partition(rows);sets,failed=build(rows,p.parent,parts);tr,va,te=[sets[k] for k in ['train','calibration','test']]
    t=time.perf_counter();m=PairedCNN().fit(tr['x'],tr['y'],va['x'],va['y']);elapsed=time.perf_counter()-t
    vs=m.forward(va['x'])[0];thresholds=np.unique(np.r_[0,vs,1])
    candidates=[]
    for threshold in thresholds:
        pred=vs>threshold;score=metrics(va['y'],pred);fpr=float(pred[va['y']==0].mean())
        if fpr<=.05:candidates.append((score['f1'] or 0,float(threshold)))
    _,threshold=max(candidates);ps=m.forward(te['x'])[0];result=metrics(te['y'],ps>threshold);result['auroc']=float(roc_auc_score(te['y'],ps));result['normal_alarm_rate']=float((ps[te['y']==0]>threshold).mean())
    report={'kind':'정합 + 64px 대응 ROI CNN 합성 실험','created_at':core.now(),'production_ready':False,'split_counts':{k:len(v) for k,v in parts.items()},'pair_counts':{k:len(s['y']) for k,s in sets.items()},'excluded_pair_count':len(failed),'alignment_rejected_count':sum(f['reason']!='unchanged' for f in failed),'unchanged_variant_count':sum(f['reason']=='unchanged' for f in failed),'threshold':threshold,'threshold_rule':'validation F1 maximum subject to FPR <= 5%; tie higher threshold','test':result,'training_seconds':elapsed,'best_epoch':m.best_epoch,'history':m.history,'test_unit':'ROI pair, not image; synthetic labels only','limitations':['4개 고대비 ROI만 시험; 전체 이미지 결함 검출 아님','이동 정합만 지원, 회전·원근·크기 변화 지원 안함','동일 시안의 합성 변형으로 실제 사진 평가 아님','같은 원본 파생 ROI는 독립 표본 아님','정합 실패 제외 후 지표; 실패 수 별도 기록','기존 시험 시안 재사용; 신규 독립 평가 필요']}
    joblib.dump({'model':m,'threshold':threshold,'production_ready':False},out/'model.joblib');np.savez_compressed(out/'weights.npz',**m.p)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(out/'dataset.json').write_text(json.dumps({'source_rows':rows,'splits':parts,'pairs':{k:s['meta'] for k,s in sets.items()},'alignment_failures':failed,'test_scores':ps.tolist()},ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
