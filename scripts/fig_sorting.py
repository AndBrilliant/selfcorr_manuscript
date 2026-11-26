#!/usr/bin/env python3
"""Figure: change in criterion (x) and sensitivity (y) for every manipulation, pooled, with 95%
task-clustered intervals. Input built by scripts/fig_sorting_points.py from the analysis JSON files.
Usage: python scripts/fig_sorting.py data/mbpp_v2/analysis/fig_sorting_points.json fig_sorting.pdf
(also writes a .png alongside)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pts = json.load(open(sys.argv[1])); out = sys.argv[2]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
col = {"prompt": "#257de0", "attr": "#f47230", "assert": "#00b894"}
mk = {"prompt": "o", "attr": "s", "assert": "^"}
nm = {"prompt": "Prompt (steelman vs. self-check)", "attr": "Attribution sentence", "assert": "Assert added"}
fig, ax = plt.subplots(figsize=(6.3, 2.9))
ax.axhline(0, color="#9c9c9c", lw=0.6, zorder=0); ax.axvline(0, color="#9c9c9c", lw=0.6, zorder=0)
num = {"Steelman, convention unstated": ("a", 7, -10), "Steelman, error visible in prose": ("b", 7, 3),
       "Steelman, assert shown": ("c", 6, 4), "Steelman, assert seen by generator": ("d", -10, 4),
       '"You wrote this"': ("e", -11, 3), '"The user wrote this"': ("f", 6, -10), "Placebo assert": ("g", 6, 4)}
seen = set()
for label, g, dc, dcci, dd, ddci in pts:
    if label == "Reference assert":
        ax.annotate("", xy=(0.64, 0.61), xytext=(0.50, 0.43),
                    arrowprops=dict(arrowstyle="-|>", color=col[g], lw=1.4, mutation_scale=10))
        ax.text(0.49, 0.43, "h: reference assert,\n$\\Delta d'=+3.59$, $\\Delta c=+1.63$\n(off scale)", ha="right", va="center", fontsize=7, color="#222222")
        if g not in seen:
            ax.plot([], [], marker=mk[g], ls="", color=col[g], label=nm[g]); seen.add(g)
        continue
    ax.errorbar(dc, dd, xerr=[[dc - dcci[0]], [dcci[1] - dc]], yerr=[[dd - ddci[0]], [ddci[1] - dd]],
                fmt=mk[g], ms=6, color=col[g], ecolor=col[g], elinewidth=0.8, capsize=0, mec="white", mew=0.8,
                alpha=1, label=nm[g] if g not in seen else None, zorder=3)
    seen.add(g)
    n, dx, dy = num[label]
    ax.annotate(n, (dc, dd), xytext=(dx, dy), textcoords="offset points", fontsize=8, fontweight="bold", color="#222222")
ax.set_xlim(-0.42, 0.65); ax.set_ylim(-0.9, 0.62)
ax.set_xlabel("Change in criterion $\\Delta c$ (stricter $\\rightarrow$)")
ax.set_ylabel("Change in sensitivity $\\Delta d'$")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(False)
ax.legend(loc="lower right", frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
