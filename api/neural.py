"""Small NumPy neural autoencoders, with explicit backpropagation and Adam."""
import numpy as np
from PIL import Image

NAMES=['CNN-AE','RNN-AE','MLP-AE']
def image_patches(a):
    h,w=a.shape;out=[]
    for y in range(4):
        for x in range(4):
            tile=a[y*h//4:(y+1)*h//4,x*w//4:(x+1)*w//4]
            im=Image.fromarray(tile.astype('float32'),mode='F').resize((16,16),Image.Resampling.BILINEAR)
            out.append(np.asarray(im,dtype=float))
    return np.array(out)

class Autoencoder:
    def __init__(self,name,seed=42):
        if name not in NAMES:raise ValueError(name)
        self.name=name;rng=np.random.default_rng(seed);self.p={}
        def weight(k,a,b):self.p[k]=rng.normal(0,np.sqrt(1/a),(a,b))
        if name=='CNN-AE':weight('conv',9,4);self.p['cb']=np.zeros(4);weight('enc',196,16)
        elif name=='RNN-AE':weight('input',16,16);weight('recur',16,16)
        else:weight('enc',256,16)
        self.p['eb']=np.zeros(16);weight('dec',16,256);self.p['db']=np.zeros(256)
    def forward(self,x):
        p=self.p;c={'x':x};flat=x.reshape(len(x),256)
        if self.name=='CNN-AE':
            windows=np.lib.stride_tricks.sliding_window_view(x,(3,3),axis=(1,2))[:,::2,::2].reshape(len(x),49,9)
            conv=windows@p['conv']+p['cb'];f=np.maximum(conv,0).reshape(len(x),196)
            z=np.tanh(f@p['enc']+p['eb']);c.update(windows=windows,conv=conv,f=f)
        elif self.name=='RNN-AE':
            states=[np.zeros((len(x),16))]
            for t in range(16):states.append(np.tanh(x[:,t]@p['input']+states[-1]@p['recur']+p['eb']))
            z=states[-1];c['states']=states
        else:
            z=np.tanh(flat@p['enc']+p['eb']);c['f']=flat
        out=1/(1+np.exp(-np.clip(z@p['dec']+p['db'],-40,40)));c.update(z=z,out=out)
        return out.reshape(x.shape),c
    def gradients(self,x):
        out,c=self.forward(x);p=self.p;g={k:np.zeros_like(v) for k,v in p.items()}
        loss=float(np.mean((out-x)**2));dy=(2*(out-x)/x.size).reshape(len(x),256)*c['out']*(1-c['out'])
        g['dec']=c['z'].T@dy;g['db']=dy.sum(0);dz=dy@p['dec'].T
        if self.name=='RNN-AE':
            for t in reversed(range(16)):
                dh=dz*(1-c['states'][t+1]**2)
                g['input']+=x[:,t].T@dh;g['recur']+=c['states'][t].T@dh;g['eb']+=dh.sum(0);dz=dh@p['recur'].T
        else:
            dh=dz*(1-c['z']**2);g['enc']=c['f'].T@dh;g['eb']=dh.sum(0)
            if self.name=='CNN-AE':
                dc=(dh@p['enc'].T).reshape(len(x),49,4)*(c['conv']>0)
                g['conv']=np.einsum('bpi,bpj->ij',c['windows'],dc);g['cb']=dc.sum((0,1))
        return loss,g
    def fit(self,train,validation,epochs=80,batch_size=32):
        rng=np.random.default_rng(42);m={k:np.zeros_like(v) for k,v in self.p.items()};v={k:np.zeros_like(v) for k,v in self.p.items()};step=0;best=float('inf');saved=None;history=[];stale=0
        for epoch in range(epochs):
            for ids in np.array_split(rng.permutation(len(train)),max(1,int(np.ceil(len(train)/batch_size)))):
                _,g=self.gradients(train[ids]);step+=1
                norm=np.sqrt(sum(np.sum(a*a) for a in g.values()));scale=min(1,5/max(norm,1e-12))
                for k in self.p:
                    grad=g[k]*scale;m[k]=.9*m[k]+.1*grad;v[k]=.999*v[k]+.001*grad*grad
                    self.p[k]-=.001*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            tl=float(np.mean((self.forward(train)[0]-train)**2));vl=float(np.mean((self.forward(validation)[0]-validation)**2));history.append({'epoch':epoch+1,'train_mse':tl,'validation_mse':vl})
            if vl<best-1e-7:best=vl;saved={k:a.copy() for k,a in self.p.items()};self.best_epoch=epoch+1;stale=0
            else:stale+=1
            if stale>=12:break
        self.p=saved;self.history=history;return self
    def score(self,x):
        error=((self.forward(x)[0]-x)**2).mean((1,2));return float(np.sort(error)[-4:].mean())
