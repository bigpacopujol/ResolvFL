from __future__ import annotations
from typing import Hashable, Mapping
import torch
import torch.nn.functional as F
from .taxonomy import Taxonomy


def exact_path_nll(
    logits_by_node: Mapping[Hashable, torch.Tensor],
    observed_node: Hashable,
    taxonomy: Taxonomy,
) -> torch.Tensor:
    """Exact marginalized NLL for one observed taxonomy node.

    `logits_by_node[u]` must be a 1-D tensor aligned with
    `taxonomy.children[u]`. For an internal-node observation, the loss is the sum of the
    conditional decisions along the root-to-observation path. A root
    observation has zero path loss.
    """
    path = taxonomy.path(observed_node)
    if len(path) <= 1:
        # Keep device/dtype when possible.
        ref = next(iter(logits_by_node.values()), torch.tensor(0.0))
        return ref.sum() * 0.0
    terms = []
    for parent, child in zip(path[:-1], path[1:]):
        logits = logits_by_node[parent]
        children = list(taxonomy.children[parent])
        target = children.index(child)
        terms.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=logits.device)))
    return torch.stack(terms).sum()
