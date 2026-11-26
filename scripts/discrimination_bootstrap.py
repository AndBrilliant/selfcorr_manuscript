"""Paired stratified bootstrap of the steelman-minus-self-check change in discrimination gap
Delta = tau - kappa (points), per family, on the MBPP paired set. Resamples candidates within the
T=0 and T=1 strata, keeping each candidate's two verdicts paired. 10,000 resamples, seed 20261121."""
from pathlib import Path as _P
ROOT=_P(__file__).resolve().parents[1]
OUT=ROOT/"data/mbpp_v2/analysis"
import json, random
P=[json.loads(l) for l in open(str(ROOT/'data/mbpp_v2/production/derived_records.jsonl'))]
rng=random.Random(20261121); out={}
for f in ['gpt-5.4','ds-v4-flash','glm-5.3-flash','qwen38-flash','gpt-6-luna']:
    rs=[r for r in P if r['model']==f and r['T'] in (0,1) and r['selfcheck_extraction']=='parsed' and r['steelman_extraction']=='parsed']
    p0=[(r['selfcheck_verdict']=='ACCEPT',r['steelman_verdict']=='ACCEPT') for r in rs if r['T']==0]
    p1=[(r['selfcheck_verdict']=='ACCEPT',r['steelman_verdict']=='ACCEPT') for r in rs if r['T']==1]
    def dd(a0,a1):
        k=[sum(x[i] for x in a0)/len(a0) for i in (0,1)]; t=[sum(x[i] for x in a1)/len(a1) for i in (0,1)]
        return 100*((t[1]-k[1])-(t[0]-k[0]))
    est=dd(p0,p1); bs=sorted(dd([rng.choice(p0) for _ in p0],[rng.choice(p1) for _ in p1]) for _ in range(10000))
    out[f]=dict(delta_change=round(est,2),ci95=[round(bs[250],1),round(bs[9749],1)])
    print(f,out[f])
json.dump(out,open(str(OUT/'discrimination_bootstrap.json'),'w'),indent=1)
