from __future__ import annotations
from typing import Dict, Iterable, Mapping, Sequence
import numpy as np


def double_dirichlet_partition(
    labels: Sequence[int],
    n_clients: int = 10,
    alpha_cls: float = 0.5,
    seed: int | None = None,
) -> Dict[int, list[int]]:
    """Allocate examples class-by-class using symmetric Dirichlet proportions."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    out = {k: [] for k in range(n_clients)}
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        props = rng.dirichlet(np.full(n_clients, alpha_cls))
        counts = rng.multinomial(len(idx), props)
        start = 0
        for k, n in enumerate(counts):
            out[k].extend(idx[start:start+n].tolist())
            start += n
    return out


def sample_annotation_depths(
    n_examples: int,
    available_depths: Sequence[int],
    alpha_dep: float = 0.5,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(len(available_depths), alpha_dep))
    depths = rng.choice(np.asarray(available_depths), size=n_examples, p=probs)
    return depths, probs


def controlled_deep_cohort_size(rho_nom: float, n_clients: int = 10) -> int:
    """Exact implementation convention used in the controlled overlap sweep.

    Python's built-in round is used, hence ties use bankers rounding:
    round(7.5)=8 and round(2.5)=2.
    """
    return int(round(float(rho_nom) * int(n_clients)))


def controlled_realized_overlap(rho_nom: float, n_clients: int = 10) -> float:
    return controlled_deep_cohort_size(rho_nom, n_clients) / float(n_clients)

