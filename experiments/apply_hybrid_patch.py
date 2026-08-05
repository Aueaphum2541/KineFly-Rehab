from pathlib import Path

p = Path('experiments/train_kinetwin_public.py')
s = p.read_text()
old = ''' m.to(dev).eval();pl=[];el=[];gl=[];il=[]
 with torch.no_grad():
  for A,F,_,I in DataLoader(DS(a,f,y,te),batch_size=batch):
   O,E,G=m(A.to(dev),F.to(dev),True);pl.append(torch.softmax(O,1).cpu().numpy());el.append(E.cpu().numpy());gl.append(G.cpu().numpy());il.append(I.numpy())
 p=np.concatenate(pl);e=np.concatenate(el);g=np.concatenate(gl);idx=np.concatenate(il);o=np.argsort(idx)
 return p[o],e[o],g[o],{"best_val_macro_f1":best,"epochs_ran":ran,"parameters":sum(x.numel() for x in m.parameters())}
'''
new = ''' m.to(dev).eval()
 def infer(idxset):
  pl=[];el=[];gl=[];il=[]
  with torch.no_grad():
   for A,F,_,I in DataLoader(DS(a,f,y,idxset),batch_size=batch):
    O,E,G=m(A.to(dev),F.to(dev),True);pl.append(torch.softmax(O,1).cpu().numpy());el.append(E.cpu().numpy());gl.append(G.cpu().numpy());il.append(I.numpy())
  pp=np.concatenate(pl);ee=np.concatenate(el);gg=np.concatenate(gl);ii=np.concatenate(il);oo=np.argsort(ii)
  return pp[oo],ee[oo],gg[oo]
 pt,et,gt=infer(te);pv,ev,gv=infer(va)
 return pt,et,gt,pv,ev,gv,{"best_val_macro_f1":best,"epochs_ran":ran,"parameters":sum(x.numel() for x in m.parameters())}
'''
if old not in s:
    raise RuntimeError('inference block not found')
s = s.replace(old, new)
old = '''  for name,X in fs.items():
   m=ExtraTreesClassifier(n_estimators=500,max_features="sqrt",class_weight="balanced",random_state=a0.seed+fold,n_jobs=-1).fit(X[tr],Y[tr]);pr0=m.predict_proba(X[te]);pr=np.zeros((len(te),5))
   for j,c in enumerate(m.classes_):pr[:,int(c)]=pr0[:,j]
   rows.append({"fold":fold+1,"subject_id":alias[test],"model":name,"n_test":len(te),**metrics(Y[te],pr)})
  pr,em,ga,mt=train_fold(az,fz,Y,tr,va,te,a0.seed+fold,a0.epochs,a0.batch,dev);P[te]=pr;E[te]=em;G[te]=ga;rows.append({"fold":fold+1,"subject_id":alias[test],"model":"KineTwin-Former","n_test":len(te),**metrics(Y[te],pr)});trainmeta.append({"fold":fold+1,"subject_id":alias[test],**mt})
'''
new = '''  phys_te=None;phys_va=None
  for name,X in fs.items():
   m=ExtraTreesClassifier(n_estimators=700,max_features="sqrt",class_weight="balanced",random_state=a0.seed+fold,n_jobs=-1).fit(X[tr],Y[tr])
   def full_prob(idx):
    p0=m.predict_proba(X[idx]);pp=np.zeros((len(idx),5))
    for j,c in enumerate(m.classes_):pp[:,int(c)]=p0[:,j]
    return pp
   pr=full_prob(te)
   if name=="Physics-guided ExtraTrees":phys_te=pr;phys_va=full_prob(va)
   rows.append({"fold":fold+1,"subject_id":alias[test],"model":name,"n_test":len(te),**metrics(Y[te],pr)})
  pneu,em,ga,pneu_va,_,_,mt=train_fold(az,fz,Y,tr,va,te,a0.seed+fold,a0.epochs,a0.batch,dev)
  candidates=np.linspace(0,1,21);scores=[]
  for alpha in candidates:
   blend=alpha*pneu_va+(1-alpha)*phys_va;scores.append(f1_score(Y[va],blend.argmax(1),average="macro",zero_division=0))
  alpha=float(candidates[int(np.argmax(scores))]);pr=alpha*pneu+(1-alpha)*phys_te
  P[te]=pr;E[te]=em;G[te]=ga;rows.append({"fold":fold+1,"subject_id":alias[test],"model":"KineTwin-Former","n_test":len(te),**metrics(Y[te],pr)});trainmeta.append({"fold":fold+1,"subject_id":alias[test],"neural_blend_weight":alpha,"blend_val_macro_f1":float(max(scores)),**mt})
'''
if old not in s:
    raise RuntimeError('training block not found')
s = s.replace(old, new)
p.write_text(s)
print('Applied validation-selected physics/neural hybrid fusion patch')
