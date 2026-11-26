"""Deterministic convention-tolerance audit of all incorrect MBPP candidates (no model calls).
Classifies each T=0 candidate by the most lenient check under which it passes every assert:
 type/format (list~tuple, set~sorted list, int~float, float tol 1e-6, str strip),
 + order-insensitive sequences, + alternative argument packing; else substantive.
Usage: python3 scripts/convention_audit.py            (audit all incorrect candidates)
       python3 scripts/convention_audit.py --validate (check the checker on 300 random correct candidates)"""
from pathlib import Path as _P
ROOT=_P(__file__).resolve().parents[1]
OUT=ROOT/"data/mbpp_v2/analysis"
import json, sys, os, subprocess, tempfile, collections, math
root=str(ROOT)
tasks={t['task_id']:t for t in map(json.loads,open(f'{root}/data/MBPP.jsonl'))}
cands=[c for c in map(json.loads,open(f'{root}/data/mbpp_v2/production/candidate_records.jsonl')) if c['T']==0]
runner=str(ROOT/'scripts/convention_audit_runner.py')
if '--validate' in sys.argv:
    import random
    c1=[c for c in map(json.loads,open(f'{root}/data/mbpp_v2/production/candidate_records.jsonl')) if c['T']==1]
    rnd=random.Random(7); samp=rnd.sample(c1,300); w=tempfile.mkdtemp(); n_strict=n_parsed=0
    for c in samp:
        t=tasks[c['task_id']]; jp=os.path.join(w,'job.json')
        json.dump({'code':c['code'],'test':t['test'],'entry_point':t['entry_point']},open(jp,'w'))
        r=subprocess.run([sys.executable,'-I',runner,jp],capture_output=True,text=True,timeout=20,cwd=w)
        A=json.loads(r.stdout.strip().splitlines()[-1]).get('asserts',[])
        n_strict+=bool(A) and all(a['strict'] for a in A); n_parsed+=bool(A) and all(a['parsed'] for a in A)
    res={'sample_seed':7,'n_correct_sampled':300,'pass_strict_all_asserts':n_strict,'all_asserts_parsed':n_parsed}
    json.dump(res,open(str(OUT/'convention_audit_validation.json'),'w'),indent=1); print(res); sys.exit()
out={}; work=tempfile.mkdtemp()
for i,c in enumerate(cands):
    t=tasks[c['task_id']]; jp=os.path.join(work,'job.json')
    json.dump({'code':c['code'],'test':t['test'],'entry_point':t['entry_point']},open(jp,'w'))
    try:
        r=subprocess.run([sys.executable,'-I',runner,jp],capture_output=True,text=True,timeout=20,cwd=work)
        line=(r.stdout.strip().splitlines() or ['{}'])[-1]; d=json.loads(line) if line.startswith('{') else {'crash':True}
    except subprocess.TimeoutExpired: d={'timeout':True}
    except Exception: d={'crash':True}
    A=d.get('asserts')
    if not A: cls='substantive'
    elif all(a['strict'] for a in A): cls='passes_strict'   # e.g. guard-blocked in production
    elif all(a['strict'] or a['type'] for a in A): cls='type_format'
    elif all(a['strict'] or a['type'] or a['order'] for a in A): cls='order'
    elif all(a['strict'] or a['type'] or a['order'] or a['args'] for a in A): cls='arg_packing'
    else: cls='substantive'
    out[(c['model'],c['task_id'])]=cls
json.dump({f'{m}|{t}':v for (m,t),v in out.items()},open(str(OUT/'convention_audit.json'),'w'),indent=0)
print('incorrect candidates',len(out)); print(collections.Counter(out.values()))
