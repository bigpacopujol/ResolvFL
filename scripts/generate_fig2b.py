import matplotlib.pyplot as plt
from _registry import load_rows, select, save_figure

rows = select(load_rows(), dataset='CIFAR-100', experiment='cohort_overlap_sweep',
              method_or_endpoint='ResolvFL-Tree', metric='leaf_accuracy_difference')
rows.sort(key=lambda r: float(r['condition'].split('=')[1]), reverse=True)
x = [float(r['condition'].split('=')[1]) for r in rows]
y = [float(r['mean']) for r in rows]
sd = [float(r['sd']) for r in rows]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.errorbar(x, y, yerr=sd, marker='o', capsize=3)
ax.axhline(0, linestyle='--', linewidth=1)
ax.set_xlabel('Realized active-cohort overlap (Jaccard)')
ax.set_ylabel('ResolvFL - Tree (percentage points)')
ax.set_xticks(x); ax.invert_xaxis(); ax.grid(alpha=.35)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
save_figure(fig, 'fig_overlap_sweep')
