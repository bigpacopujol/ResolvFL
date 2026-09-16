from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
labels=["CIFAR: Tree - Flat","CIFAR: FedGrain - Tree","iNat-Aves: Tree - Flat","iNat-Aves: FedGrain - Tree","ImageNet: Tree - Flat","ImageNet: FedGrain - Tree"]
means=np.array([0.12,0.22,9.17,2.23,6.84,-0.13])
sds=np.array([1.09,0.64,2.72,3.96,4.07,1.05])
y=np.arange(len(labels))
fig,ax=plt.subplots(figsize=(7.6,4.4))
ax.errorbar(means,y,xerr=sds,fmt="o",capsize=3)
ax.axvline(0,linestyle="--",linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Leaf-accuracy difference (percentage points)")
ax.invert_yaxis(); ax.grid(axis="x",alpha=.35)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(ROOT / f"figures/fig_paired_differences.{ext}", dpi=300 if ext=="png" else None, bbox_inches="tight")
