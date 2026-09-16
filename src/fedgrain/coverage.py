from __future__ import annotations
from typing import Dict, Hashable, Mapping
from .taxonomy import Taxonomy


def empirical_supervision_counts(
    label_histogram: Mapping[Hashable, int], taxonomy: Taxonomy
) -> Dict[Hashable, int]:
    """Count finite-dataset examples that supervise each internal node.

    An observed label supervises u iff the label lies strictly below u.
    """
    counts = {}
    for u in taxonomy.internal_nodes:
        strict_desc = set(taxonomy.descendants(u, include_self=False))
        counts[u] = int(sum(int(n) for v, n in label_histogram.items() if v in strict_desc))
    return counts


def empirical_coverage_flags(
    client_histograms: Mapping[Hashable, Mapping[Hashable, int]], taxonomy: Taxonomy
) -> Dict[Hashable, bool]:
    total = {u: 0 for u in taxonomy.internal_nodes}
    for hist in client_histograms.values():
        c = empirical_supervision_counts(hist, taxonomy)
        for u, n in c.items():
            total[u] += n
    return {u: n > 0 for u, n in total.items()}


def maximal_uncovered_roots(flags: Mapping[Hashable, bool], taxonomy: Taxonomy):
    roots = []
    for u in taxonomy.internal_nodes:
        if flags.get(u, False):
            continue
        internal_anc = [a for a in taxonomy.ancestors(u) if a in taxonomy.internal_nodes]
        if all(flags.get(a, True) for a in internal_anc):
            roots.append(u)
    return roots
