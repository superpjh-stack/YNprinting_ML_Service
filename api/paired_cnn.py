"""Paired 64px ROI CNN with bounded translation registration."""
import numpy as np
from scipy import ndimage

def align(reference,query,max_shift=8):
    if reference.shape!=query.shape:return None,{'accepted':False,'reason':'shape_mismatch'}
    a=reference-reference.mean();b=query-query.mean();f=np.fft.fft2(a)*np.conj(np.fft.fft2(b));cross=np.fft.ifft2(f/(np.abs(f)+1e-12)).real
    pos=np.array(np.unravel_index(cross.argmax(),cross.shape));shape=np.array(a.shape);shift=np.where(pos>shape//2,pos-shape,pos)
    if np.any(np.abs(shift)>max_shift):return None,{'accepted':False,'reason':'shift_out_of_range'}
    moved=ndimage.shift(query,shift,order=0,mode='constant',cval=1.);valid=ndimage.shift(np.ones_like(query),shift,order=0,mode='constant',cval=0)>0
    ar=reference[valid];br=moved[valid];corr=float(np.corrcoef(ar,br)[0,1]) if ar.std()>1e-6 and br.std()>1e-6 else 0.
    ok=bool(corr>=.75 and valid.mean()>=.75)
    # Fill unsupported border with reference; exclude it from comparisons.
    moved[~valid]=reference[~valid]
    return moved,{'accepted':ok,'reason':'ok' if ok else 'low_correlation','shift':shift.tolist(),'correlation':corr,'valid_fraction':float(valid.mean())}

def pair_tensor(ref,query):return np.stack([ref,query,np.abs(ref-query)],axis=-1)

class PairedCNN:
    def __init__(self,channels=6,kernel=3,seed=42):
        if channels<1 or kernel not in (3,5,7):raise ValueError('invalid CNN configuration')
        rng=np.random.default_rng(seed)
        self.p={'conv':rng.normal(0,.1,(kernel*kernel*3,channels)),'cb':np.zeros(channels),'head':rng.normal(0,.1,(channels*2,1)),'hb':np.zeros(1)}
    def forward(self,x):
        kernel=int(np.sqrt(self.p['conv'].shape[0]/3))
        windows=np.lib.stride_tricks.sliding_window_view(x,(kernel,kernel),axis=(1,2))[:,::2,::2].transpose(0,1,2,4,5,3).reshape(len(x),-1,kernel*kernel*3)
        z=windows@self.p['conv']+self.p['cb'];relu=np.maximum(z,0);maxidx=relu.argmax(axis=1)
        f=np.concatenate([relu.mean(axis=1),relu.max(axis=1)],axis=1)
        logits=(f@self.p['head']+self.p['hb']).ravel();prob=1/(1+np.exp(-np.clip(logits,-40,40)))
        return prob,(windows,z,relu,maxidx,f,logits)
    def gradients(self,x,y):
        prob,(windows,z,relu,maxidx,f,logits)=self.forward(x);loss=float(np.mean(np.logaddexp(0,logits)-y*logits));d=((prob-y)/len(y))[:,None]
        g={'head':f.T@d,'hb':d.sum(0)};df=d@self.p['head'].T
        channels=self.p['cb'].size
        dr=np.broadcast_to(df[:,:channels,None].transpose(0,2,1)/relu.shape[1],relu.shape).copy()
        for b in range(len(x)):dr[b,maxidx[b],np.arange(channels)]+=df[b,channels:]
        dz=dr*(z>0);g['conv']=np.einsum('bpi,bpj->ij',windows,dz);g['cb']=dz.sum((0,1))
        return loss,g
    def fit(self,x,y,vx,vy,epochs=30,learning_rate=.003,batch_size=16):
        if epochs<1 or learning_rate<=0 or batch_size<1:raise ValueError("invalid training configuration")
        rng=np.random.default_rng(42);m={k:np.zeros_like(v) for k,v in self.p.items()};v={k:np.zeros_like(a) for k,a in self.p.items()};step=0;best=np.inf;self.history=[]
        for epoch in range(epochs):
            for ids in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/batch_size))):
                _,g=self.gradients(x[ids],y[ids]);step+=1
                for k in self.p:
                    m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2;self.p[k]-=learning_rate*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            loss=self.gradients(vx,vy)[0];self.history.append({'epoch':epoch+1,'validation_bce':loss})
            if loss<best:best=loss;saved={k:a.copy() for k,a in self.p.items()};self.best_epoch=epoch+1
        self.p=saved;return self
