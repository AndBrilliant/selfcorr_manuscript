"""kappa, E and I(T;A|G) recomputed with T=0 restricted to candidates that fail even under convention tolerance."""
from pathlib import Path as _P
ROOT=_P(__file__).resolve().parents[1]
OUT=ROOT/"data/mbpp_v2/analysis"
import json, math, collections
aud=json.load(open(str(OUT/'convention_audit.json')))
prod=[json.loads(l) for l in open(str(ROOT/'data/mbpp_v2/production/derived_records.jsonl'))]
fams=['ds-v4-flash','glm-5.3-flash','qwen38-flash','gpt-5.4','gpt-6-luna']
res={}
for f in fams:
    rs=[r for r in prod if r['model']==f and r['T'] in (0,1) and r['selfcheck_extraction']=='parsed' and r['steelman_extraction']=='parsed']
    allT0=[r for r in rs if r['T']==0]; sub=[r for r in allT0 if aud[f"{f}|{r['task_id']}"]=='substantive']
    T1=[r for r in rs if r['T']==1]; res[f]={'n_T0':len(allT0),'n_sub':len(sub)}
    cl=collections.Counter(aud[f"{f}|{r['task_id']}"] for r in allT0); res[f]['classes']=dict(cl)
    for arm in ['selfcheck','steelman']:
        k=sum(r[f'{arm}_verdict']=='ACCEPT' for r in allT0)/len(allT0)
        ks=sum(r[f'{arm}_verdict']=='ACCEPT' for r in sub)/len(sub)
        kc=[r for r in allT0 if aud[f"{f}|{r['task_id']}"]!='substantive']
        kconv=sum(r[f'{arm}_verdict']=='ACCEPT' for r in kc)/len(kc) if kc else None
        t=sum(r[f'{arm}_verdict']=='ACCEPT' for r in T1)/len(T1)
        res[f][arm]=dict(kappa=round(100*k,1),kappa_sub=round(100*ks,1),kappa_conv=round(100*kconv,1) if kconv is not None else None,E=round(math.log2(t/k),3),E_sub=round(math.log2(t/ks),3))
    print(f,res[f])
json.dump(res,open(str(OUT/'convention_kappa.json'),'w'),indent=1)
