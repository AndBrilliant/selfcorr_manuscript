"""Calibrated framework tests (paper Sec. IV-E). The second part (naive-LR comparators)
gives the full-determinism reference (2E) reported in Sec. IV-E.
(1) The five moderators (scripts/mbpp_v2_moderator_spec.md) as candidate Z: plug-in I(T;A|G,Z_m) against a within-stratum permutation null
    (A permuted within each Z_m stratum, which preserves A-Z and T-Z and breaks T-A given Z).
    Same for unconditional I(T;A|G) with a global permutation null.
(2) Naive-vs-empirical unanimous LR: overstatement relative to the full-determinism null (2*E_1),
    and a naive comparator using the product of the three round-specific likelihood ratios."""
from pathlib import Path as _P
ROOT=_P(__file__).resolve().parents[1]
OUT=ROOT/"data/mbpp_v2/analysis"
import json, math, random, collections, importlib.util
spec=importlib.util.spec_from_file_location('mod',str(ROOT/'scripts/mbpp_v2_moderator_analysis.py')); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
D=str(ROOT/'data/mbpp_v2')
prod={(r['model'],r['task_id']):r for r in map(json.loads,open(f'{D}/production/derived_records.jsonl'))}
rep={(r['model'],r['task_id']):r for r in map(json.loads,open(f'{D}/repeated_eval/repeated_records.jsonl'))}
cand={(c['model'],c['task_id']):c for c in map(json.loads,open(f'{D}/production/candidate_records.jsonl'))}
tasks={t['task_id']:t for t in map(json.loads,open(str(ROOT/'data/MBPP.jsonl')))}
fams=['ds-v4-flash','glm-5.3-flash','qwen38-flash','gpt-5.4','gpt-6-luna']
rng=random.Random(20261121); NPERM=2000
def mi(a,t):
    n=len(a); c=collections.Counter(zip(a,t)); ca=collections.Counter(a); ct=collections.Counter(t)
    return sum(k/n*math.log2((k/n)/((ca[x]/n)*(ct[y]/n))) for (x,y),k in c.items())
def cmi(a,t,z):
    g=collections.defaultdict(list)
    for i,zz in enumerate(z): g[zz].append(i)
    n=len(a); return sum(len(ix)/n*mi([a[i] for i in ix],[t[i] for i in ix]) for ix in g.values())
rows=[]
for k,r in prod.items():
    if r['T'] in (0,1) and r['selfcheck_extraction']=='parsed' and r['steelman_extraction']=='parsed' and k in cand:
        rows.append(dict(model=k[0],task_id=k[1],T=r['T'],sc=int(r['selfcheck_verdict']=='ACCEPT'),st=int(r['steelman_verdict']=='ACCEPT'),code=cand[k]['code']))
mv={(r['model'],r['task_id']):mod.moderator_values(r,tasks[r['task_id']]) for r in rows}
pw=mod.tertiles([v['prompt_words'] for v in mv.values()]); cl=mod.tertiles([v['code_lines'] for v in mv.values()])
def zval(r,m):
    v=mv[(r['model'],r['task_id'])][m]
    if m=='prompt_words': return 'lo' if v<=pw[0] else ('mid' if v<=pw[1] else 'hi')
    if m=='code_lines': return 'lo' if v<=cl[0] else ('mid' if v<=cl[1] else 'hi')
    return str(v)
out={'moderators':{},'naive':{}}
for f in fams:
    fr=[r for r in rows if r['model']==f]; T=[r['T'] for r in fr]
    for arm in ['sc','st']:
        A=[r[arm] for r in fr]; obs=mi(A,T)
        null=[mi(rng.sample(A,len(A)),T) for _ in range(NPERM)]; mu=sum(null)/NPERM
        p=(1+sum(x>=obs for x in null))/(NPERM+1)
        cell={'I':round(1000*obs,1),'I_null_mean':round(1000*mu,2),'I_corrected':round(1000*(obs-mu),1),'p':round(p,4),'mods':{}}
        for m in mod.MODERATORS:
            Z=[zval(r,m) for r in fr]; o=cmi(A,T,Z)
            idx=collections.defaultdict(list)
            for i,zz in enumerate(Z): idx[zz].append(i)
            nl=[]
            for _ in range(NPERM):
                Ap=A[:]
                for ix in idx.values():
                    vals=[A[i] for i in ix]; rng.shuffle(vals)
                    for i,v in zip(ix,vals): Ap[i]=v
                nl.append(cmi(Ap,T,Z))
            mu2=sum(nl)/NPERM; p2=(1+sum(x>=o for x in nl))/(NPERM+1)
            cell['mods'][m]={'CMI':round(1000*o,1),'null_mean':round(1000*mu2,2),'corrected':round(1000*(o-mu2),1),'p':round(p2,4),'strata':len(idx)}
        out['moderators'][f'{f}|{arm}']=cell
        print(f,arm,'I',cell['I'],'corr',cell['I_corrected'],'p',cell['p'],{m:(v['corrected'],v['p']) for m,v in cell['mods'].items()})
# naive LR
for f in fams:
    for arm in ['selfcheck','steelman']:
        R=[]
        for k,r in prod.items():
            if k[0]!=f or r['T'] not in (0,1) or r['selfcheck_extraction']!='parsed' or r['steelman_extraction']!='parsed': continue
            q=rep.get(k)
            if not q or q.get(f'{arm}_r2_extraction')!='parsed' or q.get(f'{arm}_r3_extraction')!='parsed': continue
            R.append(((int(r[f'{arm}_verdict']=='ACCEPT'),int(q[f'{arm}_r2_verdict']=='ACCEPT'),int(q[f'{arm}_r3_verdict']=='ACCEPT')),r['T']))
        P0=[v for v,t in R if t==0]; P1=[v for v,t in R if t==1]
        E=[math.log2((sum(v[i] for v in P1)/len(P1))/(sum(v[i] for v in P0)/len(P0))) for i in range(3)]
        emp=math.log2((sum(sum(v)==3 for v in P1)/len(P1))/(sum(sum(v)==3 for v in P0)/len(P0)))
        naive_cube=3*E[0]; naive_prod=sum(E)
        out['naive'][f'{f}|{arm}']=dict(E1=round(E[0],3),E_rounds=[round(x,3) for x in E],naive_cube=round(naive_cube,3),naive_product=round(naive_prod,3),
            empirical=round(emp,3),over_cube=round(naive_cube-emp,3),over_product=round(naive_prod-emp,3),
            determinism_null=round(2*E[0],3),frac_of_determinism=round((naive_cube-emp)/(2*E[0]),2) if E[0]>0 else None)
        print(f,arm,out['naive'][f'{f}|{arm}'])
json.dump(out,open(str(OUT/'theory_calibrated.json'),'w'),indent=1)
