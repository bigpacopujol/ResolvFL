from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
labels = ["CIFAR\nvehicles_1", "CIFAR\nsmall_mammals", "CIFAR\nflowers", "iNat-Aves\nPetroica", "ImageNet\n5-leaf subtree"]
reference = np.array([40.47, 19.00, 40.60, 56.94, 65.87])
removed = np.array([8.33, 2.67, 14.20, 15.28, 19.07])
x=np.arange(len(labels)); w=0.38
fig,ax=plt.subplots(figsize=(8.6,4.8))
ax.bar(x-w/2, reference, w, label="Reference")
ax.bar(x+w/2, removed, w, label="Coverage removed", hatch="///")
ax.set_ylabel("Target leaf accuracy (%)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0,72); ax.grid(axis="y", alpha=.35)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(ROOT / f"figures/fig_target_subtree_violation.{ext}", dpi=300 if ext=="png" else None, bbox_inches="tight")
