from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
x=np.array([1.00,0.80,0.50,0.20,0.10])
y=np.array([-0.07,0.34,0.57,0.62,0.87])
s=np.array([0.07,0.30,0.37,1.07,0.44])
fig,ax=plt.subplots(figsize=(6.5,4.2))
ax.errorbar(x,y,yerr=s,marker="o",capsize=3)
ax.axhline(0,linestyle="--",linewidth=1)
ax.set_xlabel("Realized active-cohort overlap")
ax.set_ylabel("FedGrain - Tree (points)")
ax.set_xticks(x)
ax.invert_xaxis()
ax.grid(alpha=.35)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(ROOT / f"figures/fig_overlap_sweep.{ext}", dpi=300 if ext=="png" else None, bbox_inches="tight")
