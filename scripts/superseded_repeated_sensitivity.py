"""Repeated-verdict quantities with superseded candidate copies excluded (paper Sec. IV-G).

A repeated-evaluation call judged a superseded copy when its prompt hash differs from the
production call's prompt hash for the same (model, task, arm). Those (model, task) cells
are dropped, and the repeated-verdict endpoints are recomputed on the rest:
  I(T; A_{1:3} | G) in millibits; naive-minus-empirical unanimous log2 LR, with the naive
  benchmark as the cube of the first-call ratio and as the product of the three
  round-specific ratios, each with a stratified bootstrap 95% interval (2,000 resamples,
  seed 20261121, as in theory_extra.py).
Computed after data collection, in response to review.
Writes data/mbpp_v2/analysis/superseded_repeated_sensitivity.json.
"""
from pathlib import Path
import collections, json, math, random

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/mbpp_v2"
L = lambda p: [json.loads(l) for l in open(D / p) if l.strip()]
prod = {(r['model'], r['task_id']): r for r in L('production/derived_records.jsonl')}
rep = {(r['model'], r['task_id']): r for r in L('repeated_eval/repeated_records.jsonl')}
ev = {e['event_id']: e for e in L('production/events.jsonl')}
canon = {}
for k, d in prod.items():
    for arm in ('selfcheck', 'steelman'):
        e = ev.get(d.get(f'{arm}_event_id'))
        if e:
            canon[(k, arm)] = e['prompt_sha256']
superseded = set()
for e in L('repeated_eval/events.jsonl'):
    k = (e['model'], e['task_id']); c = canon.get((k, e['stage'].split('_')[0]))
    if c and c != e['prompt_sha256']:
        superseded.add(k)
fams = ['ds-v4-flash', 'glm-5.3-flash', 'qwen38-flash', 'gpt-5.4', 'gpt-6-luna']


def mi_multi(rows):
    n = len(rows); c = collections.Counter(rows); pT = sum(t for _, t in rows) / n
    ca = collections.Counter(a for a, _ in rows); tot = 0
    for (a, t), k in c.items():
        pt = pT if t == 1 else 1 - pT
        tot += k / n * math.log2((k / n) / ((ca[a] / n) * pt))
    return tot


def lr(P1, P0, f):
    return math.log2((sum(f(v) for v in P1) / len(P1)) / (sum(f(v) for v in P0) / len(P0)))


def over(R, kind):
    P0 = [v for v, t in R if t == 0]; P1 = [v for v, t in R if t == 1]
    emp = lr(P1, P0, lambda v: sum(v) == 3)
    if kind == 'cube':
        naive = 3 * lr(P1, P0, lambda v: v[0])
    else:
        naive = sum(lr(P1, P0, lambda v, i=i: v[i]) for i in range(3))
    return naive - emp


random.seed(20261121)
out = {'superseded_cells': len(superseded), 'cells': {}}
for f in fams:
    for arm in ['selfcheck', 'steelman']:
        R = []
        for k, r in prod.items():
            if k[0] != f or k in superseded or r['T'] not in (0, 1) or r['selfcheck_extraction'] != 'parsed' or r['steelman_extraction'] != 'parsed':
                continue
            q = rep.get(k)
            if not q or q.get(f'{arm}_r2_extraction') != 'parsed' or q.get(f'{arm}_r3_extraction') != 'parsed':
                continue
            R.append(((int(r[f'{arm}_verdict'] == 'ACCEPT'), int(q[f'{arm}_r2_verdict'] == 'ACCEPT'),
                       int(q[f'{arm}_r3_verdict'] == 'ACCEPT')), r['T']))
        P0 = [x for x in R if x[1] == 0]; P1 = [x for x in R if x[1] == 1]
        cell = {'n': len(R), 'I_A123_millibits': round(1000 * mi_multi(R), 1)}
        for kind in ('cube', 'product'):
            d = over(R, kind)
            bs = sorted(over([random.choice(P0) for _ in P0] + [random.choice(P1) for _ in P1], kind) for _ in range(2000))
            cell[f'over_{kind}'] = round(d, 3); cell[f'over_{kind}_ci95'] = [round(bs[50], 3), round(bs[1949], 3)]
        out['cells'][f'{f}|{arm}'] = cell
        print(f, arm, cell)
c = out['cells'].values()
out['summary'] = {k: {'positive': sum(v[f'over_{k}'] > 0 for v in c), 'ci_excludes_zero': sum(v[f'over_{k}_ci95'][0] > 0 for v in c)}
                  for k in ('cube', 'product')}
out['I_A123_range_millibits'] = [min(v['I_A123_millibits'] for v in c), max(v['I_A123_millibits'] for v in c)]
print(out['summary'], out['I_A123_range_millibits'])
(D / 'analysis/superseded_repeated_sensitivity.json').write_text(json.dumps(out, indent=1))
