#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, random, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

URLS=[
 "https://raw.githubusercontent.com/Pranav-Rastogi/barbell-lift/master/pml-training.csv",
 "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"]
CLASSES=list("ABCDE")
CLASS_NAMES={"A":"Correct execution","B":"Elbows displaced forward","C":"Incomplete lifting phase",
 "D":"Incomplete lowering phase","E":"Hip/trunk thrust compensation"}
ARM=[f"gyros_arm_{a}" for a in "xyz"]+[f"accel_arm_{a}" for a in "xyz"]
FORE=[f"gyros_forearm_{a}" for a in "xyz"]+[f"accel_forearm_{a}" for a in "xyz"]
META=["user_name","num_window","classe","raw_timestamp_part_1","raw_timestamp_part_2"]

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def sha256(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def download(p):
 if p.exists() and p.stat().st_size>1_000_000:return "cached"
 last=None
 for u in URLS:
  try:
   req=urllib.request.Request(u,headers={"User-Agent":"KineTwin-Former/1.0"})
   with urllib.request.urlopen(req,timeout=180) as r,open(p,"wb") as f:
    while b:=r.read(1<<20):f.write(b)
   if p.stat().st_size<1_000_000:raise RuntimeError("small download")
   return u
  except Exception as e:
   last=e
   if p.exists():p.unlink()
 raise RuntimeError(last)

def resample(x,L):
 if len(x)==L:return x.astype("float32")
 a=np.linspace(0,1,len(x)); b=np.linspace(0,1,L)
 return np.stack([np.interp(b,a,x[:,j]) for j in range(x.shape[1])],1).astype("float32")

@dataclass
class Rec:
 arm:np.ndarray; fore:np.ndarray; y:int; label:str; subject:str; sid:str; window:int; length:int

def load_data(path,L):
 df=pd.read_csv(path,usecols=META+ARM+FORE,low_memory=False)
 for c in ARM+FORE:df[c]=pd.to_numeric(df[c],errors="coerce")
 df[ARM+FORE]=df.groupby("user_name",group_keys=False)[ARM+FORE].apply(lambda g:g.interpolate(limit_direction="both"))
 df[ARM+FORE]=df[ARM+FORE].fillna(df[ARM+FORE].median()).fillna(0)
 df=df.sort_values(["user_name","raw_timestamp_part_1","raw_timestamp_part_2","num_window"])
 subs=sorted(df.user_name.astype(str).unique()); alias={s:f"S{i+1}" for i,s in enumerate(subs)}
 rec=[]
 for (s,w,c),g in df.groupby(["user_name","num_window","classe"],sort=False):
  if c not in CLASSES or len(g)<5:continue
  rec.append(Rec(resample(g[ARM].to_numpy("float32"),L),resample(g[FORE].to_numpy("float32"),L),
   CLASSES.index(c),c,str(s),alias[str(s)],int(w),len(g)))
 if len(rec)<100:raise RuntimeError(f"Only {len(rec)} windows")
 return rec,df,alias

def arrays(rec):
 return (np.stack([r.arm for r in rec]),np.stack([r.fore for r in rec]),np.array([r.y for r in rec]),
  np.array([r.subject for r in rec]),np.array([r.sid for r in rec]))

def normalize(a,f,tr):
 z=np.concatenate([a[tr],f[tr]]).reshape(-1,6); mu=z.mean(0); sd=z.std(0);sd[sd<1e-6]=1
 return ((a-mu)/sd).astype("float32"),((f-mu)/sd).astype("float32"),mu,sd

def stats(x):
 return np.concatenate([x.mean(1),x.std(1),x.min(1),x.max(1),np.median(x,1),
  np.quantile(x,.25,axis=1),np.quantile(x,.75,axis=1),np.sqrt((x*x).mean(1)),
  (np.diff(x,axis=1)**2).mean(1),np.diff(x,axis=1).mean(1)],1)

def feature_sets(a,f):
 fa,ff,fr=stats(a),stats(f),stats(f-a); corr=[]
 for j in range(6):
  x=a[:,:,j]-a[:,:,j].mean(1,keepdims=True);y=f[:,:,j]-f[:,:,j].mean(1,keepdims=True)
  corr.append((x*y).sum(1)/(np.sqrt((x*x).sum(1)*(y*y).sum(1))+1e-8))
 return {"Arm-only ExtraTrees":fa,"Forearm-only ExtraTrees":ff,
  "Early-fusion ExtraTrees":np.c_[fa,ff],"Physics-guided ExtraTrees":np.c_[fa,ff,fr,np.stack(corr,1)]}

class DS(Dataset):
 def __init__(self,a,f,y,idx):self.a=torch.from_numpy(a[idx]);self.f=torch.from_numpy(f[idx]);self.y=torch.from_numpy(y[idx]).long();self.i=torch.from_numpy(idx)
 def __len__(self):return len(self.y)
 def __getitem__(self,k):return self.a[k],self.f[k],self.y[k],self.i[k]

class KTF(nn.Module):
 def __init__(self,L,d=48,h=4,nc=5):
  super().__init__();self.ap=nn.Linear(6,d);self.fp=nn.Linear(6,d);self.rp=nn.Linear(6,d);self.pos=nn.Parameter(torch.zeros(1,L,d));nn.init.trunc_normal_(self.pos,std=.02)
  def enc():return nn.TransformerEncoder(nn.TransformerEncoderLayer(d,h,2*d,.15,batch_first=True,norm_first=True,activation="gelu"),1)
  self.ae=enc();self.fe=enc();self.af=nn.MultiheadAttention(d,h,.15,batch_first=True);self.fa=nn.MultiheadAttention(d,h,.15,batch_first=True)
  self.gate=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,d),nn.Sigmoid())
  self.cal=nn.Sequential(nn.Linear(12,d),nn.GELU(),nn.LayerNorm(d));self.fuse=enc();self.norm=nn.LayerNorm(d)
  self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(.15),nn.Linear(d,nc))
 def forward(self,a,f,ret=False):
  A=self.ae(self.ap(a)+self.pos);F=self.fe(self.fp(f)+self.pos);AC,_=self.af(A,F,F,need_weights=False);FC,_=self.fa(F,A,A,need_weights=False);R=self.rp(f-a)
  G=self.gate(torch.cat([AC.mean(1),FC.mean(1),R.mean(1)],1)).unsqueeze(1);Z=G*(A+AC)+(1-G)*(F+FC)+R
  k=max(2,a.shape[1]//5);C=self.cal(torch.cat([a[:,:k].mean(1),f[:,:k].mean(1)],1)).unsqueeze(1);E=self.norm(self.fuse(torch.cat([C,Z],1))[:,0]);O=self.head(E)
  return (O,E,G.squeeze(1)) if ret else O

def train_fold(a,f,y,tr,va,te,seed,epochs,batch,dev):
 seed_all(seed);m=KTF(a.shape[1]).to(dev);cl=np.unique(y[tr]);cw=compute_class_weight("balanced",classes=cl,y=y[tr]);W=torch.ones(5)
 for c,w in zip(cl,cw):W[int(c)]=float(w)
 loss=nn.CrossEntropyLoss(weight=W.to(dev),label_smoothing=.03);opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=1e-3);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-5)
 tl=DataLoader(DS(a,f,y,tr),batch_size=batch,shuffle=True);vl=DataLoader(DS(a,f,y,va),batch_size=batch);best=-1;state=None;pat=0;ran=0
 for ep in range(epochs):
  m.train()
  for A,F,Y,_ in tl:
   A,F,Y=A.to(dev),F.to(dev),Y.to(dev);opt.zero_grad();L=loss(m(A,F),Y);L.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  sch.step();m.eval();T=[];P=[]
  with torch.no_grad():
   for A,F,Y,_ in vl:P+=m(A.to(dev),F.to(dev)).argmax(1).cpu().tolist();T+=Y.tolist()
  v=f1_score(T,P,average="macro",zero_division=0);ran=ep+1
  if v>best+1e-4:best=v;state={k:x.detach().cpu().clone() for k,x in m.state_dict().items()};pat=0
  else:pat+=1
  if ep>=8 and pat>=7:break
 if state:m.load_state_dict(state)
 m.to(dev).eval();pl=[];el=[];gl=[];il=[]
 with torch.no_grad():
  for A,F,_,I in DataLoader(DS(a,f,y,te),batch_size=batch):
   O,E,G=m(A.to(dev),F.to(dev),True);pl.append(torch.softmax(O,1).cpu().numpy());el.append(E.cpu().numpy());gl.append(G.cpu().numpy());il.append(I.numpy())
 p=np.concatenate(pl);e=np.concatenate(el);g=np.concatenate(gl);idx=np.concatenate(il);o=np.argsort(idx)
 return p[o],e[o],g[o],{"best_val_macro_f1":best,"epochs_ran":ran,"parameters":sum(x.numel() for x in m.parameters())}

def metrics(y,p):
 q=p.argmax(1);d={"accuracy":accuracy_score(y,q),"balanced_accuracy":balanced_accuracy_score(y,q),"macro_f1":f1_score(y,q,average="macro",zero_division=0),"weighted_f1":f1_score(y,q,average="weighted",zero_division=0),"kappa":cohen_kappa_score(y,q)}
 try:d["macro_auc_ovr"]=roc_auc_score(y,p,multi_class="ovr",average="macro")
 except:d["macro_auc_ovr"]=float("nan")
 return {k:float(v) for k,v in d.items()}

def remap(y,mode):return (y!=0).astype(int) if mode=="binary" else np.array([0,1,2,2,1])[y]
def agg(p,mode):return np.c_[p[:,0],p[:,1:].sum(1)] if mode=="binary" else np.c_[p[:,0],p[:,1]+p[:,4],p[:,2]+p[:,3]]
def ece(y,p,b=10):
 c=p.max(1);z=(p.argmax(1)==y);e=0
 for lo,hi in zip(np.linspace(0,1,b+1)[:-1],np.linspace(0,1,b+1)[1:]):
  m=(c>lo)&(c<=hi)
  if m.any():e+=m.mean()*abs(z[m].mean()-c[m].mean())
 return float(e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=Path("benchmark_output"));ap.add_argument("--epochs",type=int,default=30);ap.add_argument("--seq-len",type=int,default=48);ap.add_argument("--batch",type=int,default=64);ap.add_argument("--seed",type=int,default=20260805);a0=ap.parse_args();out=a0.output;out.mkdir(parents=True,exist_ok=True)
 path=out/"pml-training.csv";src=download(path);rec,raw,alias=load_data(path,a0.seq_len);A,F,Y,S,SID=arrays(rec);subs=sorted(np.unique(S));dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 rows=[];pred=[];P=np.zeros((len(rec),5),"float32");E=np.zeros((len(rec),48),"float32");G=np.zeros((len(rec),48),"float32");trainmeta=[]
 order=["Arm-only ExtraTrees","Forearm-only ExtraTrees","Early-fusion ExtraTrees","Physics-guided ExtraTrees","KineTwin-Former"]
 for fold,test in enumerate(subs):
  rem=[s for s in subs if s!=test];val=rem[fold%len(rem)];te=np.flatnonzero(S==test);va=np.flatnonzero(S==val);tr=np.flatnonzero((S!=test)&(S!=val));az,fz,mu,sd=normalize(A,F,tr);fs=feature_sets(az,fz)
  for name,X in fs.items():
   m=ExtraTreesClassifier(n_estimators=500,max_features="sqrt",class_weight="balanced",random_state=a0.seed+fold,n_jobs=-1).fit(X[tr],Y[tr]);pr0=m.predict_proba(X[te]);pr=np.zeros((len(te),5))
   for j,c in enumerate(m.classes_):pr[:,int(c)]=pr0[:,j]
   rows.append({"fold":fold+1,"subject_id":alias[test],"model":name,"n_test":len(te),**metrics(Y[te],pr)})
  pr,em,ga,mt=train_fold(az,fz,Y,tr,va,te,a0.seed+fold,a0.epochs,a0.batch,dev);P[te]=pr;E[te]=em;G[te]=ga;rows.append({"fold":fold+1,"subject_id":alias[test],"model":"KineTwin-Former","n_test":len(te),**metrics(Y[te],pr)});trainmeta.append({"fold":fold+1,"subject_id":alias[test],**mt})
  rel=fz[te]-az[te]
  for k,i in enumerate(te):pred.append({"index":int(i),"fold":fold+1,"subject_id":alias[test],"window_id":rec[i].window,"true_label":int(Y[i]),"predicted_label":int(pr[k].argmax()),**{f"p_{CLASSES[j]}":float(pr[k,j]) for j in range(5)},"confidence":float(pr[k].max()),"arm_gyro_energy":float((az[i,:,:3]**2).mean()),"arm_accel_energy":float((az[i,:,3:]**2).mean()),"fore_gyro_energy":float((fz[i,:,:3]**2).mean()),"fore_accel_energy":float((fz[i,:,3:]**2).mean()),"rel_gyro_energy":float((rel[k,:,:3]**2).mean()),"rel_accel_energy":float((rel[k,:,3:]**2).mean()),"mean_gate":float(ga[k].mean())})
  print(f"fold {fold+1} {alias[test]} macro-F1={rows[-1]['macro_f1']:.3f}",flush=True)
 fold=pd.DataFrame(rows);fold.to_csv(out/"fold_metrics.csv",index=False);pd.DataFrame(pred).sort_values("index").to_csv(out/"predictions.csv",index=False);pd.DataFrame(trainmeta).to_csv(out/"training_meta.csv",index=False)
 met=["accuracy","balanced_accuracy","macro_f1","weighted_f1","kappa","macro_auc_ovr"];summary=[]
 for name in order:
  d=fold[fold.model==name];r={"model":name}
  for c in met:r[c+"_mean"]=d[c].mean();r[c+"_std"]=d[c].std(ddof=1)
  summary.append(r)
 sm=pd.DataFrame(summary);sm.to_csv(out/"metrics_summary.csv",index=False);q=P.argmax(1);pre,re,f1,sup=precision_recall_fscore_support(Y,q,labels=np.arange(5),zero_division=0);yb=label_binarize(Y,classes=np.arange(5));auc=[roc_auc_score(yb[:,j],P[:,j]) for j in range(5)]
 pd.DataFrame({"class_index":range(5),"class_label":CLASSES,"precision":pre,"recall":re,"f1":f1,"support":sup,"auc":auc}).to_csv(out/"class_metrics.csv",index=False);pd.DataFrame(confusion_matrix(Y,q),index=CLASSES,columns=CLASSES).to_csv(out/"confusion_counts.csv");pd.DataFrame(confusion_matrix(Y,q,normalize="true"),index=CLASSES,columns=CLASSES).to_csv(out/"confusion_normalized.csv")
 hier={"five":metrics(Y,P)}
 for mode in ["binary","taxonomy"]:hier[mode]=metrics(remap(Y,mode),agg(P,mode))
 (out/"hierarchical_metrics.json").write_text(json.dumps(hier,indent=2));np.save(out/"embeddings.npy",E);np.save(out/"probabilities.npy",P);np.save(out/"labels.npy",Y);np.save(out/"subject_ids.npy",SID)
 pd.DataFrame({"subject_id":SID,"class_label":[CLASSES[x] for x in Y],"window_id":[r.window for r in rec],"original_length":[r.length for r in rec]}).to_csv(out/"window_metadata.csv",index=False)
 best=sm[sm.model!="KineTwin-Former"].sort_values("macro_f1_mean",ascending=False).iloc[0].model;pf=fold[fold.model=="KineTwin-Former"].sort_values("subject_id").macro_f1.to_numpy();bf=fold[fold.model==best].sort_values("subject_id").macro_f1.to_numpy()
 try:ws,wp=wilcoxon(pf,bf)
 except:ws,wp=0,1
 manifest={"dataset":{"name":"UCI WLE pml-training public partition","source":src,"sha256":sha256(path),"raw_samples":len(raw),"windows":len(rec),"subjects":len(subs),"sequence_length":a0.seq_len,"class_names":CLASS_NAMES,"subject_aliases":alias},"training":{"seed":a0.seed,"epochs":a0.epochs,"batch":a0.batch,"device":str(dev),"torch":torch.__version__,"protocol":"LOSO; one additional validation subject"},"statistics":{"best_baseline":best,"wilcoxon_statistic":float(ws),"wilcoxon_p":float(wp),"median_macro_f1_difference":float(np.median(pf-bf)),"ece":ece(Y,P)}}
 (out/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2));prop=sm[sm.model=="KineTwin-Former"].iloc[0];result={"macro_f1_mean":prop.macro_f1_mean,"macro_f1_std":prop.macro_f1_std,"accuracy_mean":prop.accuracy_mean,"balanced_accuracy_mean":prop.balanced_accuracy_mean,"binary_macro_f1":hier["binary"]["macro_f1"],"taxonomy_macro_f1":hier["taxonomy"]["macro_f1"],"ece":manifest["statistics"]["ece"],"best_baseline":best,"wilcoxon_p":wp};(out/"RESULTS.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
 path.unlink(missing_ok=True)
if __name__=="__main__":main()
