from __future__ import annotations
from typing import Hashable, Mapping, Sequence
import numpy as np
from .taxonomy import Taxonomy

METRIC_REGISTRY = {
    "leaf_accuracy": "v1.0.0",
    "marginal_prefix_accuracy": "v1.0.0",
    "conditional_head_accuracy": "v1.0.0",
    "raw_internal_node_prediction": "v1.0.0",
}


def tree_leaf_posterior(head_probs: Mapping[Hashable, Sequence[float]], taxonomy: Taxonomy):
    out = {}
    for leaf in taxonomy.leaves:
        p = 1.0
        path = taxonomy.path(leaf)
        for parent, child in zip(path[:-1], path[1:]):
            children = list(taxonomy.children[parent])
            p *= float(head_probs[parent][children.index(child)])
        out[leaf] = p
    return out


def flat_leaf_posterior(flat_probs: Mapping[Hashable, float], taxonomy: Taxonomy):
    vals = np.array([float(flat_probs[l]) for l in taxonomy.leaves], dtype=float)
    denom = vals.sum()
    if denom <= 0:
        raise ValueError("Flat leaf mass is zero; cannot renormalize")
    vals /= denom
    return dict(zip(taxonomy.leaves, vals.tolist()))


def argmax_dict(scores: Mapping[Hashable, float]):
    return max(scores, key=scores.get)


def marginal_node_probability(node: Hashable, head_probs, taxonomy: Taxonomy) -> float:
    if node == taxonomy.root:
        return 1.0
    p = 1.0
    path = taxonomy.path(node)
    for parent, child in zip(path[:-1], path[1:]):
        children = list(taxonomy.children[parent])
        p *= float(head_probs[parent][children.index(child)])
    return p


def marginal_prefix_prediction(candidates, head_probs, taxonomy: Taxonomy):
    return argmax_dict({u: marginal_node_probability(u, head_probs, taxonomy) for u in candidates})


def conditional_head_prediction(parent, head_probs, taxonomy: Taxonomy):
    probs = np.asarray(head_probs[parent], dtype=float)
    return list(taxonomy.children[parent])[int(np.argmax(probs))]


def raw_internal_node_prediction(flat_probs: Mapping[Hashable, float], taxonomy: Taxonomy) -> bool:
    return argmax_dict(flat_probs) in set(taxonomy.internal_nodes)

