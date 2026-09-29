#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, time, urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader

URLS=[
 "https://raw.githubusercontent.com/Pranav-Rastogi/barbell-lift/master/pml-training.csv",
 "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"]
CLASSES=list("ABCDE")
ARM=[f"gyros_arm_{a}" for a in "xyz"]+[f"accel_arm_{a}" for a in "xyz"]
FORE=[f"gyros_forearm_{a}" for a in "xyz"]+[f"accel_forearm_{a}" for a in "xyz"]
META=["user_name","num_window","classe","raw_timestamp_part_1","raw_timestamp_part_2"]
VARIANTS=["arm_only","forearm_only","dual_no_cross","cross_no_relative","full_no_calibration","full_kinetwin"]
LABELS={
 "arm_only":"Upper-arm only",
 "forearm_only":"Forearm only",
 "dual_no_cross":"Dual stream, no cross-attention",
 "cross_no_relative":"Cross-attention, no relative branch",
 "full_no_calibration":"Cross-attention + relative branch",
 "full_kinetwin":"Full KineTwin-Former"}

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

def download(p):
 if p.exists() and p.stat().st_size>1_000_000:return "cached"
 err=None
 for u in URLS:
  try:
   req=urllib.request.Request(u,headers={"User-Agent":"KineTwin-Former-Extended/1.0"})
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

def resample(x,L):
 if len(x)==L:return x.astype("float32")
 a=np.linspace(0,1,len(x));b=np.linspace(0,1,L)
 return np.stack([np.interp(b,a,x[:,j]) for j in range(x.shape[1])],1).astype("float32")

@dataclass
class Window:
 arm:np.ndarray; fore:np.ndarray; y:int; subject:str; sid:str; window:int

def load_data(path,L):
 df=pd.read_csv(path,usecols=META+ARM+FORE,low_memory=False)
 for c in ARM+FORE:df[c]=pd.to_numeric(df[c],errors="coerce")
 df[ARM+FORE]=df.groupby("user_name",group_keys=False)[ARM+FORE].apply(lambda g:g.interpolate(limit_direction="both"))
 df[ARM+FORE]=df[ARM+FORE].fillna(df[ARM+FORE].median()).fillna(0)
 df=df.sort_values(["user_name","raw_timestamp_part_1","raw_timestamp_part_2","num_window"])
 subs=sorted(df.user_name.astype(str).unique());alias={s:f"S{i+1}" for i,s in enumerate(subs)}
 rows=[]
 for (s,w,c),g in df.groupby(["user_name","num_window","classe"],sort=False):
  if c not in CLASSES or len(g)<5:continue
  rows.append(Window(resample(g[ARM].to_numpy("float32"),L),
                     resample(g[FORE].to_numpy("float32"),L),
                     CLASSES.index(c),str(s),alias[str(s)],int(w)))
 return rows,df,alias

def arrays(rows):
 A=np.stack([r.arm for r in rows]);F=np.stack([r.fore for r in rows])
 Y=np.array([r.y for r in rows]);S=np.array([r.subject for r in rows]);SID=np.array([r.sid for r in rows])
 return A,F,Y,S,SID

def normalize(A,F,tr):
 z=np.concatenate([A[tr],F[tr]],axis=0).reshape(-1,A.shape[-1])
 mu=z.mean(0);sd=z.std(0);sd[sd<1e-6]=1
 return ((A-mu)/sd).astype("float32"),((F-mu)/sd).astype("float32")

class DS(Dataset):
 def __init__(self,A,F,Y,idx):
  self.A=torch.from_numpy(A[idx]);self.F=torch.from_numpy(F[idx]);self.Y=torch.from_numpy(Y[idx]).long()
 def __len__(self):return len(self.Y)
 def __getitem__(self,i):return self.A[i],self.F[i],self.Y[i]

def encoder(d,h):
 return nn.TransformerEncoder(nn.TransformerEncoderLayer(d,h,2*d,.15,batch_first=True,norm_first=True,activation="gelu"),1)

class VariantNet(nn.Module):
 def __init__(self,L,variant,d=48,h=4,nc=5):
  super().__init__();self.variant=variant;self.L=L
  self.ap=nn.Linear(6,d);self.fp=nn.Linear(6,d);self.rp=nn.Linear(6,d)
  self.pos=nn.Parameter(torch.zeros(1,L,d));nn.init.trunc_normal_(self.pos,std=.02)
  self.ae=encoder(d,h);self.fe=encoder(d,h)
  self.af=nn.MultiheadAttention(d,h,.15,batch_first=True)
  self.fa=nn.MultiheadAttention(d,h,.15,batch_first=True)
  self.gate=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,d),nn.Sigmoid())
  self.cal=nn.Sequential(nn.Linear(12,d),nn.GELU(),nn.LayerNorm(d))
  self.fuse=encoder(d,h);self.norm=nn.LayerNorm(d)
  head_in=2*d if variant=="dual_no_cross" else d
  self.head=nn.Sequential(nn.Linear(head_in,d),nn.GELU(),nn.Dropout(.15),nn.Linear(d,nc))
 def forward(self,A,F):
  if self.variant=="arm_only":
   H=self.ae(self.ap(A)+self.pos);E=self.norm(H.mean(1));return self.head(E)
  if self.variant=="forearm_only":
   H=self.fe(self.fp(F)+self.pos);E=self.norm(H.mean(1));return self.head(E)
  HA=self.ae(self.ap(A)+self.pos);HF=self.fe(self.fp(F)+self.pos)
  if self.variant=="dual_no_cross":
   return self.head(torch.cat([HA.mean(1),HF.mean(1)],1))
  AC,_=self.af(HA,HF,HF,need_weights=False);FC,_=self.fa(HF,HA,HA,need_weights=False)
  if self.variant=="cross_no_relative":
   Z=.5*(HA+AC)+.5*(HF+FC)
  else:
   R=self.rp(F-A);G=self.gate(torch.cat([AC.mean(1),FC.mean(1),R.mean(1)],1)).unsqueeze(1)
   Z=G*(HA+AC)+(1-G)*(HF+FC)+R
  if self.variant=="full_kinetwin":
   k=max(2,A.shape[1]//5);C=self.cal(torch.cat([A[:,:k].mean(1),F[:,:k].mean(1)],1)).unsqueeze(1)
   E=self.norm(self.fuse(torch.cat([C,Z],1))[:,0])
  else:
   E=self.norm(self.fuse(Z).mean(1))
  return self.head(E)

def metric_dict(y,p):
 q=p.argmax(1)
 out={"accuracy":accuracy_score(y,q),"balanced_accuracy":balanced_accuracy_score(y,q),
      "macro_f1":f1_score(y,q,average="macro",zero_division=0)}
 try:out["macro_auc"]=roc_auc_score(y,p,multi_class="ovr",average="macro")
 except:out["macro_auc"]=float("nan")
 return {k:float(v) for k,v in out.items()}

def fit_model(A,F,Y,tr,va,variant,seed,epochs,batch,dev):
 seed_all(seed);m=VariantNet(A.shape[1],variant).to(dev)
 cl=np.unique(Y[tr]);cw=compute_class_weight("balanced",classes=cl,y=Y[tr]);w=torch.ones(5)
 for c,x in zip(cl,cw):w[int(c)]=float(x)
 loss=nn.CrossEntropyLoss(weight=w.to(dev),label_smoothing=.03)
 opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=1e-3)
 sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-5)
 tl=DataLoader(DS(A,F,Y,tr),batch_size=batch,shuffle=True);vl=DataLoader(DS(A,F,Y,va),batch_size=batch)
 best=-1;state=None;pat=0;ran=0
 for ep in range(epochs):
  m.train()
  for a,f,y in tl:
   a,f,y=a.to(dev),f.to(dev),y.to(dev);opt.zero_grad();L=loss(m(a,f),y);L.backward()
   nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  sch.step();m.eval();T=[];P=[]
  with torch.no_grad():
   for a,f,y in vl:
    P+=m(a.to(dev),f.to(dev)).argmax(1).cpu().tolist();T+=y.tolist()
  score=f1_score(T,P,average="macro",zero_division=0);ran=ep+1
  if score>best+1e-4:
   best=score;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};pat=0
  else:pat+=1
  if ep>=8 and pat>=7:break
 if state:m.load_state_dict(state)
 return m.to(dev).eval(),ran,best

def predict(m,A,F,Y,idx,batch,dev):
 probs=[];truth=[]
 with torch.no_grad():
  for a,f,y in DataLoader(DS(A,F,Y,idx),batch_size=batch):
   probs.append(torch.softmax(m(a.to(dev),f.to(dev)),1).cpu().numpy());truth.append(y.numpy())
 return np.concatenate(truth),np.concatenate(probs)

def interpolate_loss(X,rate,rng):
 if rate<=0:return X.copy()
 Z=X.copy()
 n,L,C=Z.shape
 for i in range(n):
  mask=rng.random(L)<rate
  mask[0]=False;mask[-1]=False
  keep=np.flatnonzero(~mask)
  miss=np.flatnonzero(mask)
  if len(miss):
   for c in range(C):Z[i,miss,c]=np.interp(miss,keep,Z[i,keep,c])
 return Z

def latency_ms(m,L,dev,reps=200):
 a=torch.zeros(1,L,6,device=dev);f=torch.zeros(1,L,6,device=dev)
 with torch.no_grad():
  for _ in range(20):m(a,f)
  vals=[]
  for _ in range(reps):
   t=time.perf_counter();m(a,f)
   if dev.type=="cuda":torch.cuda.synchronize()
   vals.append((time.perf_counter()-t)*1000)
 return float(np.median(vals)),float(np.percentile(vals,95))

def run(args):
 seed_all(args.seed);out=args.output;out.mkdir(parents=True,exist_ok=True);fig=out/"figures";fig.mkdir(exist_ok=True)
 path=out/"pml-training.csv";source=download(path);rows,raw,alias=load_data(path,args.seq_len);A,F,Y,S,SID=arrays(rows)
 subs=sorted(np.unique(S));dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 ab=[];rob=[];deploy=[]
 for fold,test in enumerate(subs):
  rem=[s for s in subs if s!=test];val=rem[fold%len(rem)]
  te=np.flatnonzero(S==test);va=np.flatnonzero(S==val);tr=np.flatnonzero((S!=test)&(S!=val))
  An,Fn=normalize(A,F,tr)
  full=None
  for vi,v in enumerate(VARIANTS):
   m,ran,bv=fit_model(An,Fn,Y,tr,va,v,args.seed+fold*100+vi,args.epochs,args.batch,dev)
   yt,p=predict(m,An,Fn,Y,te,args.batch,dev)
   ab.append({"fold":fold+1,"subject_id":alias[test],"variant":v,"label":LABELS[v],
              "n_test":len(te),"epochs":ran,"best_val_macro_f1":bv,
              "parameters":sum(x.numel() for x in m.parameters()),**metric_dict(yt,p)})
   if v=="full_kinetwin":full=m
   print(f"fold={fold+1} {alias[test]} {v} F1={ab[-1]['macro_f1']:.3f}",flush=True)
  med,p95=latency_ms(full,args.seq_len,dev)
  deploy.append({"fold":fold+1,"subject_id":alias[test],"parameters":sum(x.numel() for x in full.parameters()),
                 "median_latency_ms":med,"p95_latency_ms":p95,"device":str(dev)})
  for rate in [0,.05,.10,.20,.30]:
   rngA=np.random.default_rng(args.seed+fold*1000+int(rate*100))
   rngF=np.random.default_rng(args.seed+9999+fold*1000+int(rate*100))
   Al=interpolate_loss(An[te],rate,rngA);Fl=interpolate_loss(Fn[te],rate,rngF)
   idx=np.arange(len(te));yt,p=predict(full,Al,Fl,Y[te],idx,args.batch,dev)
   rob.append({"fold":fold+1,"subject_id":alias[test],"packet_loss_rate":rate,"n_test":len(te),**metric_dict(yt,p)})
 ab=pd.DataFrame(ab);rob=pd.DataFrame(rob);deploy=pd.DataFrame(deploy)
 ab.to_csv(out/"ablation_fold_metrics.csv",index=False);rob.to_csv(out/"robustness_fold_metrics.csv",index=False);deploy.to_csv(out/"deployment_metrics.csv",index=False)
 asum=[]
 for v,d in ab.groupby("variant",sort=False):
  r={"variant":v,"label":LABELS[v],"parameters":int(d.parameters.iloc[0])}
  for c in ["accuracy","balanced_accuracy","macro_f1","macro_auc"]:r[c+"_mean"]=d[c].mean();r[c+"_std"]=d[c].std(ddof=1)
  asum.append(r)
 asum=pd.DataFrame(asum);asum["variant"]=pd.Categorical(asum.variant,categories=VARIANTS,ordered=True);asum=asum.sort_values("variant")
 asum.to_csv(out/"ablation_summary.csv",index=False)
 rsum=[]
 for rate,d in rob.groupby("packet_loss_rate"):
  r={"packet_loss_rate":rate}
  for c in ["accuracy","balanced_accuracy","macro_f1","macro_auc"]:r[c+"_mean"]=d[c].mean();r[c+"_std"]=d[c].std(ddof=1)
  rsum.append(r)
 rsum=pd.DataFrame(rsum).sort_values("packet_loss_rate");rsum.to_csv(out/"robustness_summary.csv",index=False)
 dep={"parameters":int(deploy.parameters.iloc[0]),"median_latency_ms_mean":float(deploy.median_latency_ms.mean()),
      "median_latency_ms_std":float(deploy.median_latency_ms.std(ddof=1)),
      "p95_latency_ms_mean":float(deploy.p95_latency_ms.mean()),"device":str(dev)}
 (out/"deployment_summary.json").write_text(json.dumps(dep,indent=2))
 manifest={"dataset":{"name":"UCI Weight Lifting Exercises public training partition","source":source,"raw_samples":len(raw),
                      "exercise_windows":len(rows),"subjects":len(subs),"sequence_length":args.seq_len,"channels_per_segment":6},
           "protocol":{"evaluation":"LOSO with one additional validation subject","seed":args.seed,"epochs_max":args.epochs,
                       "batch_size":args.batch,"packet_loss":"independent packet-level temporal masking per sensor followed by within-window linear interpolation; test only; no retraining"},
           "software":{"torch":torch.__version__,"device":str(dev)}}
 (out/"extended_manifest.json").write_text(json.dumps(manifest,indent=2))
 # plots
 plt.figure(figsize=(9.2,5.5));x=np.arange(len(asum))
 plt.bar(x,asum.macro_f1_mean*100,yerr=asum.macro_f1_std*100,capsize=4)
 plt.xticks(x,asum.label,rotation=25,ha="right");plt.ylabel("Macro-F1 (%)");plt.title("Neural component ablation under subject-independent LOSO");plt.grid(axis="y",alpha=.25);plt.tight_layout()
 plt.savefig(fig/"fig07_ablation.pdf",bbox_inches="tight");plt.savefig(fig/"fig07_ablation.png",dpi=300,bbox_inches="tight");plt.close()
 plt.figure(figsize=(7.8,5.0))
 plt.errorbar(rsum.packet_loss_rate*100,rsum.macro_f1_mean*100,yerr=rsum.macro_f1_std*100,marker="o",capsize=4,label="Macro-F1")
 plt.errorbar(rsum.packet_loss_rate*100,rsum.accuracy_mean*100,yerr=rsum.accuracy_std*100,marker="s",capsize=4,label="Accuracy")
 plt.xlabel("Simulated packet loss (%)");plt.ylabel("Performance (%)");plt.title("Robustness to missing IMU packets after interpolation");plt.grid(alpha=.25);plt.legend();plt.tight_layout()
 plt.savefig(fig/"fig08_packet_loss.pdf",bbox_inches="tight");plt.savefig(fig/"fig08_packet_loss.png",dpi=300,bbox_inches="tight");plt.close()
 # tex
 def pm(m,s,scale=100):return f"{m*scale:.2f} $\\\\pm$ {s*scale:.2f}"
 rows_tex=[]
 for _,r in asum.iterrows():
  rows_tex.append(f"{r['label']} & {r['parameters']/1000:.1f}k & {pm(r.accuracy_mean,r.accuracy_std)} & {pm(r.balanced_accuracy_mean,r.balanced_accuracy_std)} & {pm(r.macro_f1_mean,r.macro_f1_std)} & {r.macro_auc_mean:.3f} $\\\\pm$ {r.macro_auc_std:.3f} \\\\\\\\")
 rob_tex=[]
 base=float(rsum.iloc[0].macro_f1_mean)
 for _,r in rsum.iterrows():
  drop=(base-r.macro_f1_mean)*100
  rob_tex.append(f"{int(round(r.packet_loss_rate*100))}\\% & {pm(r.accuracy_mean,r.accuracy_std)} & {pm(r.macro_f1_mean,r.macro_f1_std)} & {r.macro_auc_mean:.3f} $\\\\pm$ {r.macro_auc_std:.3f} & {drop:.2f} \\\\\\\\")
 full=asum[asum.variant=="full_kinetwin"].iloc[0];arm=asum[asum.variant=="arm_only"].iloc[0];fore=asum[asum.variant=="forearm_only"].iloc[0]
 dual=asum[asum.variant=="dual_no_cross"].iloc[0];cross=asum[asum.variant=="cross_no_relative"].iloc[0];rel=asum[asum.variant=="full_no_calibration"].iloc[0]
 t=rf"""
% Add after the existing Results/Interpretation subsection.
% Requires: \usepackage{{booktabs}}, \usepackage{{graphicx}}

\subsection{{Additional Experimental Setup and Reproducibility}}
To isolate architectural contributions beyond the personalization experiment,
an additional neural ablation was performed on the same public UCI Weight
Lifting Exercises partition. Only the six raw inertial channels from the
upper-arm and forearm units (three-axis angular velocity and three-axis
acceleration) were used in this experiment to match the six-axis dual-IMU
hardware abstraction. Exercise windows were resampled to {args.seq_len} time
steps. Evaluation followed subject-independent leave-one-subject-out (LOSO)
testing, with one additional participant from the remaining training subjects
used exclusively for validation and early stopping. Normalization statistics
were estimated from the training subjects only. All variants used the same
optimizer, class-weighted cross-entropy objective, maximum of {args.epochs}
epochs, batch size {args.batch}, and fixed seed {args.seed}. Reported values are
the mean and standard deviation across the six held-out participants.

\subsection{{Neural Component Ablation}}
Table~\ref{{tab:ktf_ablation}} evaluates progressively richer variants of the
proposed temporal architecture. Single-sensor models establish the information
available from either anatomical segment alone. The dual-stream model tests the
benefit of combining both sensors without cross-segment attention, after which
cross-attention and the relative arm--forearm branch are introduced
sequentially. The final configuration additionally includes the calibration
token used by KineTwin-Former.

\begin{{table*}}[t]
\centering
\caption{{Neural component ablation under subject-independent LOSO evaluation.
Values are mean $\pm$ standard deviation across six held-out participants.}}
\label{{tab:ktf_ablation}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\toprule
Configuration & Parameters & Accuracy (\%) & Balanced Acc. (\%) & Macro-F1 (\%) & Macro-AUC \\
\midrule
{chr(10).join(rows_tex)}
\bottomrule
\end{{tabular}}}}
\end{{table*}}

The full model obtained a macro-F1 of {full.macro_f1_mean*100:.2f}\%, compared
with {arm.macro_f1_mean*100:.2f}\% for the upper-arm-only model and
{fore.macro_f1_mean*100:.2f}\% for the forearm-only model. Dual-stream fusion
without cross-attention achieved {dual.macro_f1_mean*100:.2f}\%, while adding
cross-attention yielded {cross.macro_f1_mean*100:.2f}\%. Introducing the
relative arm--forearm branch produced {rel.macro_f1_mean*100:.2f}\%. These
results quantify how the two-sensor interaction and relative-motion pathway
affect cross-subject recognition rather than assuming that each architectural
component is beneficial.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig07_ablation.pdf}}
\caption{{Macro-F1 of neural ablation variants under leakage-safe
subject-independent LOSO evaluation. Error bars denote inter-subject standard
deviation.}}
\label{{fig:ktf_ablation}}
\end{{figure}}

\subsection{{Robustness to Simulated Wireless Packet Loss}}
Wireless wearable systems may contain missing packets even when the sensing
model itself is unchanged. Robustness was therefore evaluated without
retraining the full KineTwin-Former. For each held-out participant, complete
IMU packets were removed independently from the proximal and distal test
streams at rates of 5--30\%. Missing samples were reconstructed by linear
interpolation within each temporal window before inference. Training and
validation data were left unmodified, preventing the perturbation experiment
from providing additional information to the model.

\begin{{table}}[t]
\centering
\caption{{Full-model robustness to simulated packet loss. The final column is
the absolute macro-F1 reduction relative to the unperturbed test stream.}}
\label{{tab:packet_loss}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{ccccc}}
\toprule
Loss rate & Accuracy (\%) & Macro-F1 (\%) & Macro-AUC & $\Delta$F1 (pp) \\
\midrule
{chr(10).join(rob_tex)}
\bottomrule
\end{{tabular}}}}
\end{{table}}

At 20\% packet loss, macro-F1 changed from
{rsum.iloc[0].macro_f1_mean*100:.2f}\% to
{rsum[rsum.packet_loss_rate==.20].iloc[0].macro_f1_mean*100:.2f}\%, and at
30\% loss it was {rsum[rsum.packet_loss_rate==.30].iloc[0].macro_f1_mean*100:.2f}\%.
The experiment therefore characterizes degradation of the existing model
under communication loss; it does not imply robustness to arbitrary sensor
failure or prolonged disconnection.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig08_packet_loss.pdf}}
\caption{{Accuracy and macro-F1 as a function of simulated missing IMU packets.
Packet loss is applied only to held-out test streams and is followed by
within-window interpolation.}}
\label{{fig:packet_loss}}
\end{{figure}}

\subsection{{Deployment Characteristics}}
The full network contains {dep['parameters']/1000:.1f}k trainable parameters.
On the GitHub Actions {dep['device']} runner used for the reproducible
experiment, batch-one inference required
{dep['median_latency_ms_mean']:.2f} ms on average for the median latency
measurement, with a mean fold-wise 95th-percentile latency of
{dep['p95_latency_ms_mean']:.2f} ms. These measurements describe the benchmark
runner rather than the M5StickC Plus 2 microcontroller; in the present
prototype, the wearable nodes acquire and transmit IMU data while model
inference is executed off-device.

\section{{Discussion}}
The combined experiments reveal that the main challenge is not merely
within-subject movement discrimination but transfer to previously unseen
participants. The personalization results in the main experiment and the
neural ablation results are therefore complementary: the former quantifies how
a small participant-specific calibration set can reduce domain shift, whereas
the latter identifies which sensing and interaction components remain useful
when no target-subject labels are available. Importantly, improvements are not
assumed to be monotonic. The six-participant cohort is small, and participant
S1 remained substantially harder than S5 and S6 in the personalized analysis.
This heterogeneity supports uncertainty-aware calibration and longitudinal
digital-twin updating rather than a single population threshold.

\subsection{{Limitations and Future Validation}}
The public dataset is a surrogate benchmark and was not recorded using the
project's two M5StickC Plus 2 nodes. It does not provide synchronized
goniometer or optical-motion-capture elbow angles, and it contains no external
fatigue reference. Consequently, the present experiments validate
movement-quality recognition, personalization behavior, architectural
ablation, and communication robustness, but they do not constitute direct
clinical validation of ROM reconstruction or fatigue-related state estimation.
Furthermore, two arm-mounted sensors cannot directly quantify trunk
displacement. Future validation should use synchronized dual-M5StickC
recordings with a reference goniometer or motion-capture system, controlled
compensation labels, externally anchored fatigue or perceived-exertion
measurements, and a larger multi-session participant cohort.

\section{{Conclusion}}
This work presented KineTwin-Former, a dual-IMU neural kinematic digital-twin
framework that combines relative upper-arm--forearm motion, cross-segment
temporal interaction, participant-specific calibration, and safety-bounded
rehabilitation logic. The public subject-wise evaluation demonstrated that
cross-participant domain shift is substantial and that limited personalized
calibration can improve movement-quality recognition. Additional neural
ablation quantified the contribution of single- versus dual-segment sensing,
cross-attention, relative-motion modeling, and calibration-aware fusion, while
a separate packet-loss experiment measured degradation under missing wireless
samples without test-time retraining. The present results support the
framework as a reproducible proof of concept for personalized wearable
movement assessment. Direct validation of joint-angle reconstruction,
fatigue-related dynamics, and longitudinal rehabilitation adaptation remains a
necessary next step using synchronized project-specific hardware and
independent biomechanical reference measurements.
"""
 (out/"KineTwin_Additional_Results.tex").write_text(t.strip()+"\n")
 result={"full_macro_f1_mean":float(full.macro_f1_mean),"full_accuracy_mean":float(full.accuracy_mean),
         "full_macro_auc_mean":float(full.macro_auc_mean),"packet_loss_30_macro_f1":float(rsum[rsum.packet_loss_rate==.30].iloc[0].macro_f1_mean),
         "parameters":dep["parameters"],"median_latency_ms":dep["median_latency_ms_mean"]}
 (out/"EXTENDED_RESULTS.json").write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 path.unlink(missing_ok=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=Path("extended_output"))
 ap.add_argument("--seq-len",type=int,default=48);ap.add_argument("--epochs",type=int,default=24)
 ap.add_argument("--batch",type=int,default=64);ap.add_argument("--seed",type=int,default=20260929)
 run(ap.parse_args())
