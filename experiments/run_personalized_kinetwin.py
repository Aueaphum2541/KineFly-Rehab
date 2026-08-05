#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, random, urllib.request
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

URLS=[
 "https://raw.githubusercontent.com/Pranav-Rastogi/barbell-lift/master/pml-training.csv",
 "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"]
CLASSES=list("ABCDE")
CLASS_NAMES={"A":"Correct execution","B":"Elbows displaced forward","C":"Incomplete lifting phase","D":"Incomplete lowering phase","E":"Hip/trunk thrust compensation"}
ARM=["roll_arm","pitch_arm","yaw_arm","total_accel_arm"]+[f"gyros_arm_{a}" for a in "xyz"]+[f"accel_arm_{a}" for a in "xyz"]
FORE=["roll_forearm","pitch_forearm","yaw_forearm","total_accel_forearm"]+[f"gyros_forearm_{a}" for a in "xyz"]+[f"accel_forearm_{a}" for a in "xyz"]
META=["user_name","num_window","classe","raw_timestamp_part_1","raw_timestamp_part_2"]

@dataclass
class Window:
 arm:np.ndarray; fore:np.ndarray; y:int; label:str; subject:str; sid:str; window:int; order:int; length:int

def seed_all(s:int)->None:
 random.seed(s); np.random.seed(s)

def sha256(p:Path)->str:
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def download(p:Path)->str:
 if p.exists() and p.stat().st_size>1_000_000:return "cached"
 err=None
 for u in URLS:
  try:
   req=urllib.request.Request(u,headers={"User-Agent":"KineTwin-Former/2.0"})
   with urllib.request.urlopen(req,timeout=180) as r,open(p,"wb") as f:
    while True:
     b=r.read(1<<20)
     if not b:break
     f.write(b)
   if p.stat().st_size<1_000_000:raise RuntimeError("download incomplete")
   return u
  except Exception as e:
   err=e
   if p.exists():p.unlink()
 raise RuntimeError(err)

def resample(x:np.ndarray,L:int)->np.ndarray:
 if len(x)==L:return x.astype("float32")
 a=np.linspace(0,1,len(x));b=np.linspace(0,1,L)
 return np.stack([np.interp(b,a,x[:,j]) for j in range(x.shape[1])],1).astype("float32")

def load_windows(path:Path,L:int):
 df=pd.read_csv(path,usecols=META+ARM+FORE,low_memory=False)
 for c in ARM+FORE:df[c]=pd.to_numeric(df[c],errors="coerce")
 df[ARM+FORE]=df.groupby("user_name",group_keys=False)[ARM+FORE].apply(lambda g:g.interpolate(limit_direction="both"))
 df[ARM+FORE]=df[ARM+FORE].fillna(df[ARM+FORE].median()).fillna(0)
 df=df.sort_values(["user_name","raw_timestamp_part_1","raw_timestamp_part_2","num_window"])
 subjects=sorted(df.user_name.astype(str).unique());alias={s:f"S{i+1}" for i,s in enumerate(subjects)}
 rows=[]
 for (s,w,c),g in df.groupby(["user_name","num_window","classe"],sort=False):
  if c not in CLASSES or len(g)<5:continue
  order=int(g["raw_timestamp_part_1"].iloc[0])*1_000_000+int(g["raw_timestamp_part_2"].iloc[0])
  rows.append(Window(resample(g[ARM].to_numpy("float32"),L),resample(g[FORE].to_numpy("float32"),L),CLASSES.index(c),c,str(s),alias[str(s)],int(w),order,len(g)))
 if len(rows)<100:raise RuntimeError(f"Only {len(rows)} windows found")
 return rows,df,alias

def stack(rows):
 A=np.stack([r.arm for r in rows]);F=np.stack([r.fore for r in rows]);Y=np.array([r.y for r in rows]);S=np.array([r.subject for r in rows]);SID=np.array([r.sid for r in rows]);O=np.array([r.order for r in rows]);return A,F,Y,S,SID,O

def channel_fit(A,F,idx):
 X=np.concatenate([A[idx],F[idx]],axis=2).reshape(-1,A.shape[2]+F.shape[2]);mu=np.nanmedian(X,0);q1=np.nanquantile(X,.25,0);q3=np.nanquantile(X,.75,0);sc=q3-q1;sc[sc<1e-6]=np.nanstd(X[:,sc<1e-6],axis=0)+1e-6
 return mu,sc

def channel_apply(A,F,mu,sc):
 X=np.concatenate([A,F],axis=2);X=(X-mu[None,None,:])/sc[None,None,:];return X[:,:,:A.shape[2]].astype("float32"),X[:,:,A.shape[2]:].astype("float32")

def subjectwise_training_normalization(A,F,S,idx):
 An=np.zeros_like(A);Fn=np.zeros_like(F)
 for s in np.unique(S[idx]):
  ii=idx[S[idx]==s];mu,sc=channel_fit(A,F,ii);aa,ff=channel_apply(A[ii],F[ii],mu,sc);An[ii]=aa;Fn[ii]=ff
 return An,Fn

def one_channel_features(x):
 # x: [N,L,C]
 eps=1e-8
 d=np.diff(x,axis=1)
 centered=x-x.mean(1,keepdims=True)
 spec=np.abs(np.fft.rfft(centered,axis=1))**2
 spec=spec/(spec.sum(1,keepdims=True)+eps)
 freq=np.linspace(0,1,spec.shape[1],dtype=np.float32)
 lag=(centered[:,:-1]*centered[:,1:]).sum(1)/((centered[:,:-1]**2).sum(1)**.5*(centered[:,1:]**2).sum(1)**.5+eps)
 zc=(np.diff(np.signbit(centered),axis=1)!=0).mean(1)
 feats=[x.mean(1),x.std(1),x.min(1),x.max(1),np.ptp(x,axis=1),np.median(x,axis=1),np.quantile(x,.25,axis=1),np.quantile(x,.75,axis=1),np.sqrt((x*x).mean(1)),np.abs(x).mean(1),np.abs(d).mean(1),d.std(1),np.abs(d).max(1),skew(x,axis=1,bias=False,nan_policy="omit"),kurtosis(x,axis=1,bias=False,nan_policy="omit"),lag,zc,(spec*freq[None,:,None]).sum(1),-(spec*np.log(spec+eps)).sum(1),freq[np.argmax(spec,axis=1)]]
 out=np.concatenate(feats,axis=1);return np.nan_to_num(out,nan=0,posinf=0,neginf=0)

def extract_features(A,F):
 R=F-A
 base=np.concatenate([one_channel_features(A),one_channel_features(F),one_channel_features(R)],axis=1)
 corr=[]
 for j in range(A.shape[2]):
  a=A[:,:,j]-A[:,:,j].mean(1,keepdims=True);f=F[:,:,j]-F[:,:,j].mean(1,keepdims=True)
  corr.append((a*f).sum(1)/(np.sqrt((a*a).sum(1)*(f*f).sum(1))+1e-8))
 shape=np.concatenate([A[:,::4,:],F[:,::4,:],R[:,::4,:]],axis=2).reshape(len(A),-1)
 return np.concatenate([base,np.stack(corr,1),shape],axis=1).astype("float32")

def ensure_proba(p,classes,n=5):
 out=np.zeros((len(p),n),dtype=float)
 for j,c in enumerate(classes):out[:,int(c)]=p[:,j]
 out=np.clip(out,1e-9,None);out/=out.sum(1,keepdims=True);return out

def svc_proba(model,X,n=5):
 d=model.decision_function(X)
 if d.ndim==1:d=np.c_[-d,d]
 p=softmax(d,axis=1)
 classes=model.named_steps["svc"].classes_
 return ensure_proba(p,classes,n)

def fit_predict_ensemble(Xtr,ytr,Xte,seed,weights=None):
 et=ExtraTreesClassifier(n_estimators=900,max_features=.55,min_samples_leaf=1,class_weight="balanced_subsample",random_state=seed,n_jobs=-1)
 svc=Pipeline([("scale",StandardScaler()),("pca",PCA(n_components=.995,svd_solver="full",whiten=True,random_state=seed)),("svc",SVC(C=7.5,gamma="scale",class_weight="balanced",decision_function_shape="ovr",random_state=seed))])
 et.fit(Xtr,ytr,sample_weight=weights);svc.fit(Xtr,ytr,svc__sample_weight=weights)
 pe=ensure_proba(et.predict_proba(Xte),et.classes_);ps=svc_proba(svc,Xte)
 return .55*pe+.45*ps

def prototype_proba(Xcal,ycal,Xte):
 sc=StandardScaler().fit(Xcal);Z=sc.transform(Xcal);T=sc.transform(Xte)
 nc=min(24,max(2,len(Xcal)-1),Z.shape[1]);pca=PCA(n_components=nc,whiten=True,random_state=0).fit(Z);Z=pca.transform(Z);T=pca.transform(T)
 ds=[]
 for c in range(5):
  z=Z[ycal==c]
  if len(z)==0:ds.append(np.full(len(T),1e6))
  else:
   center=np.median(z,axis=0);ds.append(np.sqrt(((T-center)**2).mean(1)))
 D=np.stack(ds,1);tau=np.median(D[D<1e5])+1e-6;return softmax(-D/tau,axis=1)

def knn_proba(Xcal,ycal,Xte):
 k=min(7,max(1,len(Xcal)//10));pipe=Pipeline([("scale",StandardScaler()),("pca",PCA(n_components=min(20,len(Xcal)-1,Xcal.shape[1]),whiten=True,random_state=0)),("knn",KNeighborsClassifier(n_neighbors=k,weights="distance",p=2))]);pipe.fit(Xcal,ycal);return ensure_proba(pipe.predict_proba(Xte),pipe.named_steps["knn"].classes_)

def hierarchical_predict(Xtr,ytr,Xte,seed,weights=None):
 yb=(ytr!=0).astype(int)
 pbin=fit_predict_ensemble(Xtr,yb,Xte,seed,weights)
 p_correct=pbin[:,0] if pbin.shape[1]>=2 else 1-pbin[:,0]
 err=ytr!=0
 perr=fit_predict_ensemble(Xtr[err],ytr[err]-1,Xte,seed+100,None if weights is None else weights[err])[:,:4]
 out=np.zeros((len(Xte),5));out[:,0]=p_correct;out[:,1:]=(1-p_correct[:,None])*perr;out=np.clip(out,1e-9,None);out/=out.sum(1,keepdims=True);return out

def metrics(y,p):
 q=p.argmax(1);d={"accuracy":accuracy_score(y,q),"balanced_accuracy":balanced_accuracy_score(y,q),"macro_f1":f1_score(y,q,average="macro",zero_division=0),"weighted_f1":f1_score(y,q,average="weighted",zero_division=0)}
 try:d["macro_auc_ovr"]=roc_auc_score(y,p,multi_class="ovr",average="macro")
 except:d["macro_auc_ovr"]=float("nan")
 return {k:float(v) for k,v in d.items()}

def choose_calibration(indices,Y,O,k,classes=range(5)):
 cal=[]
 for c in classes:
  ii=indices[Y[indices]==c];ii=ii[np.argsort(O[ii])];cal.extend(ii[:k].tolist())
 cal=np.array(sorted(set(cal)),dtype=int);test=np.setdiff1d(indices,cal,assume_unique=False);return cal,test

def run(args):
 seed_all(args.seed);out=args.output;out.mkdir(parents=True,exist_ok=True);path=out/"pml-training.csv";src=download(path);rows,raw,alias=load_windows(path,args.seq_len);A,F,Y,S,SID,O=stack(rows);subjects=sorted(np.unique(S));records=[];pred_rows=[]
 protocols=[("strict_loso",0),("personalized_1shot",1),("personalized_3shot",3),("personalized_5shot",5),("personalized_10shot",10)]
 for fold,test_subject in enumerate(subjects):
  train=np.flatnonzero(S!=test_subject);held=np.flatnonzero(S==test_subject)
  # strict subject-independent evaluation
  mu,sc=channel_fit(A,F,train);An,Fn=channel_apply(A,F,mu,sc);X=extract_features(An,Fn);pte=fit_predict_ensemble(X[train],Y[train],X[held],args.seed+fold);ph=hierarchical_predict(X[train],Y[train],X[held],args.seed+fold)
  p=.65*pte+.35*ph;records.append({"fold":fold+1,"subject_id":alias[test_subject],"protocol":"strict_loso","shots_per_class":0,"n_calibration":0,"n_test":len(held),**metrics(Y[held],p)})
  for ii,pp in zip(held,p):pred_rows.append({"index":int(ii),"fold":fold+1,"subject_id":alias[test_subject],"protocol":"strict_loso","shots_per_class":0,"true_label":int(Y[ii]),"predicted_label":int(pp.argmax()),**{f"p_{CLASSES[j]}":float(pp[j]) for j in range(5)}})
  # personalized calibration protocol
  for pname,k in protocols[1:]:
   cal,te=choose_calibration(held,Y,O,k)
   if len(te)<10 or len(np.unique(Y[cal]))<5:continue
   Atr=np.zeros_like(A);Ftr=np.zeros_like(F)
   for s in np.unique(S[train]):
    ii=train[S[train]==s];m,scl=channel_fit(A,F,ii);aa,ff=channel_apply(A[ii],F[ii],m,scl);Atr[ii]=aa;Ftr[ii]=ff
   m,scl=channel_fit(A,F,cal);aa,ff=channel_apply(A[np.r_[cal,te]],F[np.r_[cal,te]],m,scl);Atr[np.r_[cal,te]]=aa;Ftr[np.r_[cal,te]]=ff
   idx_all=np.r_[train,cal];Xall=extract_features(Atr,Ftr);w=np.ones(len(idx_all));w[len(train):]=max(8.0,len(train)/(len(cal)*2.5))
   pdir=fit_predict_ensemble(Xall[idx_all],Y[idx_all],Xall[te],args.seed+1000+fold*20+k,w)
   phier=hierarchical_predict(Xall[idx_all],Y[idx_all],Xall[te],args.seed+2000+fold*20+k,w)
   pproto=prototype_proba(Xall[cal],Y[cal],Xall[te]);pknn=knn_proba(Xall[cal],Y[cal],Xall[te])
   p=.40*pdir+.20*phier+.25*pproto+.15*pknn;p=np.clip(p,1e-9,None);p/=p.sum(1,keepdims=True)
   records.append({"fold":fold+1,"subject_id":alias[test_subject],"protocol":pname,"shots_per_class":k,"n_calibration":len(cal),"n_test":len(te),**metrics(Y[te],p)})
   for ii,pp in zip(te,p):pred_rows.append({"index":int(ii),"fold":fold+1,"subject_id":alias[test_subject],"protocol":pname,"shots_per_class":k,"true_label":int(Y[ii]),"predicted_label":int(pp.argmax()),**{f"p_{CLASSES[j]}":float(pp[j]) for j in range(5)}})
   print(f"fold={fold+1} subject={alias[test_subject]} protocol={pname} macroF1={records[-1]['macro_f1']:.3f}",flush=True)
 fold_df=pd.DataFrame(records);fold_df.to_csv(out/"personalized_fold_metrics.csv",index=False);pred=pd.DataFrame(pred_rows);pred.to_csv(out/"personalized_predictions.csv",index=False)
 summary=[]
 for p,d in fold_df.groupby("protocol"):
  r={"protocol":p,"shots_per_class":int(d.shots_per_class.iloc[0]),"mean_n_test":float(d.n_test.mean())}
  for c in ["accuracy","balanced_accuracy","macro_f1","weighted_f1","macro_auc_ovr"]:r[c+"_mean"]=float(d[c].mean());r[c+"_std"]=float(d[c].std(ddof=1))
  summary.append(r)
 sm=pd.DataFrame(summary).sort_values("shots_per_class");sm.to_csv(out/"personalized_summary.csv",index=False)
 primary="personalized_5shot" if "personalized_5shot" in set(sm.protocol) else sm.sort_values("macro_f1_mean",ascending=False).iloc[0].protocol
 pp=pred[pred.protocol==primary];yt=pp.true_label.to_numpy();cols=[f"p_{c}" for c in CLASSES];pr=pp[cols].to_numpy();yp=pr.argmax(1)
 pd.DataFrame(confusion_matrix(yt,yp,labels=np.arange(5)),index=CLASSES,columns=CLASSES).to_csv(out/"primary_confusion_counts.csv");pd.DataFrame(confusion_matrix(yt,yp,labels=np.arange(5),normalize="true"),index=CLASSES,columns=CLASSES).to_csv(out/"primary_confusion_normalized.csv")
 prec,rec,f1,sup=precision_recall_fscore_support(yt,yp,labels=np.arange(5),zero_division=0);yb=label_binarize(yt,classes=np.arange(5));au=[]
 for j in range(5):
  try:au.append(roc_auc_score(yb[:,j],pr[:,j]))
  except:au.append(float("nan"))
 pd.DataFrame({"class_label":CLASSES,"precision":prec,"recall":rec,"f1":f1,"support":sup,"auc":au}).to_csv(out/"primary_class_metrics.csv",index=False)
 manifest={"dataset":{"name":"UCI Weight Lifting Exercises public training partition","source":src,"sha256":sha256(path),"raw_samples":len(raw),"exercise_windows":len(rows),"subjects":len(subjects),"sequence_length":args.seq_len,"channels_per_segment":10,"class_names":CLASS_NAMES,"subject_aliases":alias},"protocol":{"strict":"Leave-one-subject-out without target-subject labels","personalized":"Chronologically first K windows per class used for therapist-guided calibration; all remaining windows used once for testing","primary_protocol":primary,"seed":args.seed},"model":{"description":"physics-guided arm/forearm/relative feature ensemble with direct, hierarchical, prototype, and kNN personalization"}}
 (out/"personalized_manifest.json").write_text(json.dumps(manifest,indent=2))
 best=sm.sort_values("macro_f1_mean",ascending=False).iloc[0].to_dict();strict=sm[sm.protocol=="strict_loso"].iloc[0].to_dict();result={"primary_protocol":primary,"best_protocol":best["protocol"],"best_macro_f1_mean":best["macro_f1_mean"],"best_accuracy_mean":best["accuracy_mean"],"best_macro_auc_mean":best["macro_auc_ovr_mean"],"strict_macro_f1_mean":strict["macro_f1_mean"],"absolute_macro_f1_gain":best["macro_f1_mean"]-strict["macro_f1_mean"]};(out/"PERSONALIZED_RESULTS.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);path.unlink(missing_ok=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=Path("personalized_output"));ap.add_argument("--seq-len",type=int,default=64);ap.add_argument("--seed",type=int,default=20260805);run(ap.parse_args())
