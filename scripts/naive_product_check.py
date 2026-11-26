"""Naive-independence benchmark using round-specific likelihood ratios (paper Sec. IV-E).

Same candidates, same verdicts and same stratified bootstrap as theory_extra.py, but the
naive benchmark is the product of the three round-specific likelihood ratios
(log2: E_1 + E_2 + E_3) instead of the cube of the first-call ratio (3 * E_1). This
separates dependence from round-to-round differences in the marginal acceptance rates.
Computed after data collection, in response to review.
Writes data/mbpp_v2/analysis/naive_product_check.json.
"""
from pathlib import Path as _P
import json, math, random

ROOT = _P(__file__).resolve().parents[1]
OUT = ROOT / "data/mbpp_v2/analysis"
D = str(ROOT / "data/mbpp_v2")
prod = {(r['model'], r['task_id']): r for r in map(json.loads, open(f'{D}/production/derived_records.jsonl'))}
rep = {(r['model'], r['task_id']): r for r in map(json.loads, open(f'{D}/repeated_eval/repeated_records.jsonl'))}
fams = ['ds-v4-flash', 'glm-5.3-flash', 'qwen38-flash', 'gpt-5.4', 'gpt-6-luna']
random.seed(20261121)


def diff_bits(R):
    P0 = [v for v, t in R if t == 0]; P1 = [v for v, t in R if t == 1]
    naive = sum(math.log2((sum(v[i] for v in P1) / len(P1)) / (sum(v[i] for v in P0) / len(P0))) for i in range(3))
    emp = math.log2((sum(sum(v) == 3 for v in P1) / len(P1)) / (sum(sum(v) == 3 for v in P0) / len(P0)))
    return naive - emp


out = {}
for f in fams:
    for arm in ['selfcheck', 'steelman']:
        R = []
        for k, r in prod.items():
            if k[0] != f or r['T'] not in (0, 1) or r['selfcheck_extraction'] != 'parsed' or r['steelman_extraction'] != 'parsed':
                continue
            q = rep.get(k)
            if not q or q.get(f'{arm}_r2_extraction') != 'parsed' or q.get(f'{arm}_r3_extraction') != 'parsed':
                continue
            R.append(((int(r[f'{arm}_verdict'] == 'ACCEPT'), int(q[f'{arm}_r2_verdict'] == 'ACCEPT'),
                       int(q[f'{arm}_r3_verdict'] == 'ACCEPT')), r['T']))
        d = diff_bits(R)
        P0 = [x for x in R if x[1] == 0]; P1 = [x for x in R if x[1] == 1]
        bs = sorted(diff_bits([random.choice(P0) for _ in P0] + [random.choice(P1) for _ in P1]) for _ in range(2000))
        out[f'{f}|{arm}'] = dict(over_product_bits=round(d, 3), ci95=[round(bs[50], 3), round(bs[1949], 3)])
        print(f, arm, out[f'{f}|{arm}'])
pos = sum(v['over_product_bits'] > 0 for v in out.values())
sig = sum(v['ci95'][0] > 0 for v in out.values())
out['summary'] = {'positive': pos, 'ci_excludes_zero': sig, 'cells': 10}
print(out['summary'])
json.dump(out, open(str(OUT / 'naive_product_check.json'), 'w'), indent=1)
