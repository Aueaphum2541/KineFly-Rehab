#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import run_personalized_kinetwin as k

LABELS={
 "arm_only":"Upper-arm features only",
 "forearm_only":"Forearm features only",
 "dual_early":"Dual-sensor early fusion",
 "dual_relative":"Dual + relative kinematics",
 "full_personalized":"Full personalized ensemble"
}
ORDER=list(LABELS)

def feature_sets(A,F):
 arm=np.concatenate([k.one_channel_features(A),A[:,::4,:].reshape(len(A),-1)],1).astype("float32")
 fore=np.concatenate([k.one_channel_features(F),F[:,::4,:].reshape(len(F),-1)],1).astype("float32")
 dual=np.concatenate([k.one_channel_features(A),k.one_channel_features(F),
                      A[:,::4,:].reshape(len(A),-1),F[:,::4,:].reshape(len(F),-1)],1).astype("float32")
 rel=k.extract_features(A,F)
 return {"arm_only":arm,"forearm_only":fore,"dual_early":dual,"dual_relative":rel}

def run(args):
 k.seed_all(args.seed);out=args.output;out.mkdir(parents=True,exist_ok=True)
 path=out/"pml-training.csv";src=k.download(path);rows,raw,alias=k.load_windows(path,args.seq_len)
 A,F,Y,S,SID,O=k.stack(rows);subjects=sorted(np.unique(S));records=[]
 for fold,test_subject in enumerate(subjects):
  train=np.flatnonzero(S!=test_subject);held=np.flatnonzero(S==test_subject)
  cal,te=k.choose_calibration(held,Y,O,args.shots)
  Atr=np.zeros_like(A);Ftr=np.zeros_like(F)
  for s in np.unique(S[train]):
   ii=train[S[train]==s];m,sc=k.channel_fit(A,F,ii);aa,ff=k.channel_apply(A[ii],F[ii],m,sc);Atr[ii]=aa;Ftr[ii]=ff
  m,sc=k.channel_fit(A,F,cal);target=np.r_[cal,te];aa,ff=k.channel_apply(A[target],F[target],m,sc);Atr[target]=aa;Ftr[target]=ff
  idx=np.r_[train,cal];w=np.ones(len(idx));w[len(train):]=max(8.0,len(train)/(len(cal)*2.5))
  fs=feature_sets(Atr,Ftr)
  for vi,v in enumerate(["arm_only","forearm_only","dual_early","dual_relative"]):
   X=fs[v];p=k.fit_predict_ensemble(X[idx],Y[idx],X[te],args.seed+fold*100+vi,w)
   records.append({"fold":fold+1,"subject_id":alias[test_subject],"variant":v,"label":LABELS[v],
                   "n_calibration":len(cal),"n_test":len(te),**k.metrics(Y[te],p)})
  X=fs["dual_relative"]
  pdir=k.fit_predict_ensemble(X[idx],Y[idx],X[te],args.seed+5000+fold,w)
  phier=k.hierarchical_predict(X[idx],Y[idx],X[te],args.seed+6000+fold,w)
  pproto=k.prototype_proba(X[cal],Y[cal],X[te])
  pknn=k.knn_proba(X[cal],Y[cal],X[te])
  p=.40*pdir+.20*phier+.25*pproto+.15*pknn;p=np.clip(p,1e-9,None);p/=p.sum(1,keepdims=True)
  records.append({"fold":fold+1,"subject_id":alias[test_subject],"variant":"full_personalized",
                  "label":LABELS["full_personalized"],"n_calibration":len(cal),"n_test":len(te),**k.metrics(Y[te],p)})
  print(f"fold={fold+1} {alias[test_subject]} fullF1={records[-1]['macro_f1']:.3f}",flush=True)
 df=pd.DataFrame(records);df.to_csv(out/"personalized_ablation_folds.csv",index=False)
 summ=[]
 for v in ORDER:
  d=df[df.variant==v];r={"variant":v,"label":LABELS[v]}
  for c in ["accuracy","balanced_accuracy","macro_f1","weighted_f1","macro_auc_ovr"]:
   r[c+"_mean"]=d[c].mean();r[c+"_std"]=d[c].std(ddof=1)
  summ.append(r)
 sm=pd.DataFrame(summ);sm.to_csv(out/"personalized_ablation_summary.csv",index=False)
 result={"shots_per_class":args.shots,"best_variant":sm.sort_values("macro_f1_mean",ascending=False).iloc[0].variant,
         "best_macro_f1":float(sm.macro_f1_mean.max()),"full_macro_f1":float(sm[sm.variant=="full_personalized"].iloc[0].macro_f1_mean)}
 (out/"PERSONALIZED_ABLATION_RESULTS.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 path.unlink(missing_ok=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=Path("personalized_ablation_output"))
 ap.add_argument("--seq-len",type=int,default=64);ap.add_argument("--shots",type=int,default=5);ap.add_argument("--seed",type=int,default=20260929)
 run(ap.parse_args())
