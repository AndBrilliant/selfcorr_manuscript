"""(1) Bootstrap CI for log2(naive LR) - log2(empirical LR), three unanimous acceptances, repeated-evaluation subset.
(2) The five moderators (scripts/mbpp_v2_moderator_spec.md) as candidate Z: I(T;A|G) vs I(T;A|G,Z_m)."""
from pathlib import Path as _P
ROOT=_P(__file__).resolve().parents[1]
OUT=ROOT/"data/mbpp_v2/analysis"
import json, math, random, collections, sys, importlib.util
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('mod',str(ROOT/'scripts/mbpp_v2_moderator_analysis.py')); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
D=str(ROOT/'data/mbpp_v2')
prod={(r['model'],r['task_id']):r for r in map(json.loads,open(f'{D}/production/derived_records.jsonl'))}
rep={(r['model'],r['task_id']):r for r in map(json.loads,open(f'{D}/repeated_eval/repeated_records.jsonl'))}
cand={(c['model'],c['task_id']):c for c in map(json.loads,open(f'{D}/production/candidate_records.jsonl'))}
tasks={t['task_id']:t for t in map(json.loads,open(str(ROOT/'data/MBPP.jsonl')))}
fams=['ds-v4-flash','glm-5.3-flash','qwen38-flash','gpt-5.4','gpt-6-luna']
random.seed(20261121); out={'naive_vs_emp':{},'moderator_Z':{}}
def lr_bits(R):
    P0=[v for v,t in R if t==0]; P1=[v for v,t in R if t==1]
    k1=sum(v[0] for v in P0)/len(P0); t1=sum(v[0] for v in P1)/len(P1)
    k3=sum(sum(v)==3 for v in P0)/len(P0); t3=sum(sum(v)==3 for v in P1)/len(P1)
    return 3*math.log2(t1/k1)-math.log2(t3/k3)
for f in fams:
    for arm in ['selfcheck','steelman']:
        R=[]
        for k,r in prod.items():
            if k[0]!=f or r['T'] not in (0,1) or r['selfcheck_extraction']!='parsed' or r['steelman_extraction']!='parsed': continue
            q=rep.get(k)
            if not q or q.get(f'{arm}_r2_extraction')!='parsed' or q.get(f'{arm}_r3_extraction')!='parsed': continue
            R.append(((int(r[f'{arm}_verdict']=='ACCEPT'),int(q[f'{arm}_r2_verdict']=='ACCEPT'),int(q[f'{arm}_r3_verdict']=='ACCEPT')),r['T']))
        d=lr_bits(R); P0=[x for x in R if x[1]==0]; P1=[x for x in R if x[1]==1]
        bs=sorted(lr_bits([random.choice(P0) for _ in P0]+[random.choice(P1) for _ in P1]) for _ in range(2000))
        out['naive_vs_emp'][f'{f}|{arm}']=dict(diff_bits=round(d,3),ci95=[round(bs[50],3),round(bs[1949],3)])
# moderators
def H(ps): return -sum(p*math.log2(p) for p in ps if p>0)
def mi(pairs):
    n=len(pairs); c=collections.Counter(pairs); ca=collections.Counter(a for a,_ in pairs); ct=collections.Counter(t for _,t in pairs)
    return sum(k/n*math.log2((k/n)/((ca[a]/n)*(ct[t]/n))) for (a,t),k in c.items())
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
for f in fams:
    fr=[r for r in rows if r['model']==f]; o={}
    for arm in ['sc','st']:
        base=mi([(r[arm],r['T']) for r in fr]); o[arm]={'I_TA_given_G':round(1000*base,1)}
        for m in mod.MODERATORS+('joint',):
            strata=collections.defaultdict(list)
            for r in fr: strata[tuple(zval(r,x) for x in mod.MODERATORS) if m=='joint' else zval(r,m)].append(r)
            n=len(fr); cmi=sum(len(s)/n*mi([(r[arm],r['T']) for r in s]) for s in strata.values() if len(set(r['T'] for r in s))>1 and len(set(r[arm] for r in s))>1)
            hz=H([len(s)/n for s in strata.values()])
            o[arm][m]=dict(I_TA_given_GZ=round(1000*cmi,1),n_strata=len(strata))
    out['moderator_Z'][f]=o
json.dump(out,open(str(OUT/'theory_extra.json'),'w'),indent=1)
for k,v in out['naive_vs_emp'].items(): print(k,v)
for f,o in out['moderator_Z'].items():
    for arm in o: print(f,arm,o[arm]['I_TA_given_G'],{m:o[arm][m]['I_TA_given_GZ'] for m in list(mod.MODERATORS)+['joint']})
