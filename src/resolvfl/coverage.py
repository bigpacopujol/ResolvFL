from __future__ import annotations
from typing import Dict, Hashable, Mapping
from .taxonomy import Taxonomy


def empirical_supervision_counts(
    label_histogram: Mapping[Hashable, int], taxonomy: Taxonomy
) -> Dict[Hashable, int]:
    """Count finite-dataset examples that supervise each internal node.

    An observed label supervises u iff the label lies strictly below u.
    """
    # Accumulate observed-label mass bottom-up in O(|V| + |E|) time.
    # A label at u supervises its proper ancestors, never u itself.
    order = []
    stack = [taxonomy.root]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(taxonomy.children.get(u, ()))
    subtree_mass = {}
    counts = {}
    for u in reversed(order):
        children = taxonomy.children.get(u, ())
        below = sum(subtree_mass[v] for v in children)
        subtree_mass[u] = int(label_histogram.get(u, 0)) + below
        if children:
            counts[u] = below
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
    stack = [taxonomy.root]
    while stack:
        u = stack.pop()
        children = taxonomy.children.get(u, ())
        if not children:
            continue
        if not flags.get(u, False):
            roots.append(u)
        else:
            stack.extend(reversed(children))
    return roots
