"""Theory-linked quantities from the frozen MBPP production and repeated-evaluation ledgers.
Usage: python3 scripts/theory_quantities.py"""
from pathlib import Path as _P
ROOT=_P(__file__).resolve().parents[1]
OUT=ROOT/"data/mbpp_v2/analysis"
import json, sys, math, random, collections
D=str(ROOT/'data/mbpp_v2')
prod={}
for l in open(f'{D}/production/derived_records.jsonl'):
    r=json.loads(l); prod[(r['model'],r['task_id'])]=r
rep={}
for l in open(f'{D}/repeated_eval/repeated_records.jsonl'):
    r=json.loads(l); rep[(r['model'],r['task_id'])]=r
def H(p): return 0 if p in (0,1) else -(p*math.log2(p)+(1-p)*math.log2(1-p))
def mi(pairs):  # pairs of (A,T)
    n=len(pairs); c=collections.Counter(pairs); pT=sum(t for _,t in pairs)/n
    tot=0
    for (a,t),k in c.items():
        pa=sum(v for (aa,_),v in c.items() if aa==a)/n
        pt=pT if t==1 else 1-pT
        tot+=k/n*math.log2((k/n)/(pa*pt))
    return tot, H(pT)
def mi_multi(rows):  # rows: (tuple_of_A, T)
    n=len(rows); c=collections.Counter(rows); pT=sum(t for _,t in rows)/n
    ca=collections.Counter(a for a,_ in rows); tot=0
    for (a,t),k in c.items():
        pt=pT if t==1 else 1-pT; tot+=k/n*math.log2((k/n)/((ca[a]/n)*pt))
    return tot
random.seed(20261121)
fams=['ds-v4-flash','glm-5.3-flash','qwen38-flash','gpt-5.4','gpt-6-luna']
out={}
for f in fams:
    rs=[r for (m,_),r in prod.items() if m==f and r['T'] in (0,1) and r['selfcheck_extraction']=='parsed' and r['steelman_extraction']=='parsed']
    out[f]={'paired':len(rs)}
    for arm in ['selfcheck','steelman']:
        pairs=[(1 if r[f'{arm}_verdict']=='ACCEPT' else 0, r['T']) for r in rs]
        def stats(p):
            n0=sum(1 for _,t in p if t==0); n1=len(p)-n0
            k=sum(a for a,t in p if t==0)/n0; ta=sum(a for a,t in p if t==1)/n1
            return k,ta
        k,ta=stats(pairs); E=math.log2(ta/k); bound=math.log2(1/k)
        I,HT=mi(pairs)
        bsE=[];bsI=[]
        P0=[p for p in pairs if p[1]==0]; P1=[p for p in pairs if p[1]==1]
        for _ in range(2000):
            b=[random.choice(P0) for _ in P0]+[random.choice(P1) for _ in P1]
            kk,tt=stats(b); bsE.append(math.log2(tt/kk)); bsI.append(mi(b)[0])
        bsE.sort(); bsI.sort()
        out[f][arm]=dict(kappa=round(100*k,1),tau=round(100*ta,1),E=round(E,3),E_ci=[round(bsE[50],3),round(bsE[1949],3)],
            bound=round(bound,3),I_TA=round(I,4),I_ci=[round(bsI[50],4),round(bsI[1949],4)],H_T=round(HT,3),I_frac_pct=round(100*I/HT,2))
        # repeated: candidates with production verdict parsed + r2,r3 parsed in this arm
        R=[]
        for r in rs:
            q=rep.get((f,r['task_id']))
            if not q: continue
            if q.get(f'{arm}_r2_extraction')!='parsed' or q.get(f'{arm}_r3_extraction')!='parsed': continue
            v=(1 if r[f'{arm}_verdict']=='ACCEPT' else 0, 1 if q[f'{arm}_r2_verdict']=='ACCEPT' else 0, 1 if q[f'{arm}_r3_verdict']=='ACCEPT' else 0)
            R.append((v,r['T']))
        n0=sum(1 for _,t in R if t==0); n1=len(R)-n0
        k1=sum(v[0] for v,t in R if t==0)/n0; t1=sum(v[0] for v,t in R if t==1)/n1
        k3=sum(1 for v,t in R if t==0 and sum(v)==3)/n0; t3=sum(1 for v,t in R if t==1 and sum(v)==3)/n1
        naive=3*math.log2(t1/k1); emp=math.log2(t3/k3)
        inc=[mi_multi([(v[:j],t) for v,t in R]) for j in (1,2,3)]
        out[f][arm]['repeated']=dict(n=len(R),naive_bits_k3=round(naive,3),empirical_bits_k3=round(emp,3),
            naive_LR=round(2**naive,3),empirical_LR=round(2**emp,3),I_A1=round(inc[0],4),I_A12=round(inc[1],4),I_A123=round(inc[2],4))
json.dump(out,open(str(OUT/'theory_quantities.json'),'w'),indent=1)
for f in fams:
    print(f, out[f]['paired'])
    for arm in ['selfcheck','steelman']:
        a=out[f][arm]; r=a['repeated']
        print(f"  {arm:9s} k={a['kappa']} t={a['tau']} E={a['E']} {a['E_ci']} bound={a['bound']} I={a['I_TA']} {a['I_ci']} ({a['I_frac_pct']}% of H(T)={a['H_T']})")
        print(f"            rep n={r['n']} naiveLR={r['naive_LR']} empLR={r['empirical_LR']}  I1={r['I_A1']} I12={r['I_A12']} I123={r['I_A123']}")
