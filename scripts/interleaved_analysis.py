"""Interleaved rerun (exploratory): contrasts (majority vote over 3 replicates, T=0 stratum).
Usage: python3 scripts/interleaved_analysis.py   (reads data/mbpp_v2/interleaved_rerun/events.jsonl)
Note: the verdict regex accepts a verdict at the end of a prose line; 11 responses are of
that form, and treating them as malformed changes no majority verdict."""
from pathlib import Path as _P
ROOT=_P(__file__).resolve().parents[1]
OUT=ROOT/"data/mbpp_v2/analysis"
import json, re, sys, random, collections
from math import comb
ev=[json.loads(l) for l in open(ROOT/'data/mbpp_v2/interleaved_rerun/events.jsonl')]
pat=re.compile(r'VERDICT:\s*(ACCEPT|REJECT)\s*$')
cell=collections.defaultdict(list); pos=collections.defaultdict(list)
for e in ev:
    raw=(e['raw_response'] or '').strip(); m=pat.search(raw.splitlines()[-1].strip()) if raw else None
    a=1 if (m and m.group(1)=='ACCEPT') else 0   # malformed -> NO
    cell[(e['cand_family'],e['arm'],e['task_id'])].append(a)
    if e['T']==0: pos[(e['cand_family'],e['arm'])].append((e['request_index'],a))
T={(e['cand_family'],e['task_id']):e['T'] for e in ev}
def mv(f,arm): return {t:int(sum(v)>=2) for (ff,aa,t),v in cell.items() if ff==f and aa==arm}
def mcn(b,c):
    n=b+c; k=min(b,c); return min(1,2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1
random.seed(20261120); R={}
def contrast(f,x,y,label):
    X=mv(f,x);Y=mv(f,y); ts=[t for t in X if t in Y and T[(f,t)]==0]
    d=[X[t]-Y[t] for t in ts]; b=d.count(1); c=d.count(-1); est=100*sum(d)/len(d)
    bs=sorted(100*sum(random.choice(d) for _ in d)/len(d) for _ in range(10000))
    R[label]=dict(n=len(ts),kx=round(100*sum(X[t] for t in ts)/len(ts),1),ky=round(100*sum(Y[t] for t in ts)/len(ts),1),
        d=round(est,1),ci95=[round(bs[250],1),round(bs[9749],1)],ci90=[round(bs[500],1),round(bs[9499],1)],b=b,c=c,p=mcn(b,c))
for f,ev_self,ev_part in [('qwen','Qwen','GPT-6-Luna'),('luna','GPT-6-Luna','Qwen')]:
    contrast(f,'control_self','steel_self_a',f'prompt: {ev_self} judging own')
    contrast(f,'control_partner','steel_partner',f'prompt: {ev_part} judging {ev_self}')
    contrast(f,'steel_self_b','steel_self_a',f'A/A: {ev_self} steelman replicate')
# same-family interaction (steelman arm), each bracket = own-family minus partner on that family's candidates
def bracket(f):
    X=mv(f,'steel_self_a');Y=mv(f,'steel_partner'); ts=[t for t in X if t in Y and T[(f,t)]==0]; return [X[t]-Y[t] for t in ts]
bq=bracket('qwen'); bl=bracket('luna'); m=lambda d:100*sum(d)/len(d)
bs=sorted((m([random.choice(bq) for _ in bq])+m([random.choice(bl) for _ in bl]))/2 for _ in range(10000))
R['same_family']=dict(bracket_qwen=round(m(bq),1),bracket_luna=round(m(bl),1),I_same=round((m(bq)+m(bl))/2,1),
    ci95=[round(bs[250],1),round(bs[9749],1)],evaluator_leniency_qwen_minus_luna=round((m(bq)-m(bl))/2,1),
    qwen_bc=[bq.count(1),bq.count(-1)],luna_bc=[bl.count(1),bl.count(-1)])
# drift audit: acceptance on T=0 in first vs second half of the window, per arm
dr={}
for k,v in pos.items():
    v.sort(); h=len(v)//2; a=sum(x for _,x in v[:h])/h; b=sum(x for _,x in v[h:])/(len(v)-h); dr[f'{k[0]}|{k[1]}']=round(100*(b-a),1)
R['drift_second_minus_first_half_pp']=dr
json.dump(R,open(str(OUT/'interleaved_results.json'),'w'),indent=1,default=float)
for k,v in R.items(): print(k,v)
