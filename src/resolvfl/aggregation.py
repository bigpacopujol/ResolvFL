from __future__ import annotations
from typing import Mapping, Hashable


def _weighted_average(values, weights):
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Aggregation weights must have positive sum")
    out = None
    for value, weight in zip(values, weights):
        term = value * (float(weight) / total)
        out = term if out is None else out + term
    return out


def resolvfl_endpoint_average(
    local_endpoints: Mapping[Hashable, object],
    client_weights: Mapping[Hashable, float],
    supervision_fractions: Mapping[Hashable, float],
    previous_endpoint=None,
):
    """Implemented node-head aggregation: weights proportional to w_k mhat_k(u)."""
    active = [k for k in local_endpoints if supervision_fractions.get(k, 0.0) > 0]
    if not active:
        if previous_endpoint is None:
            raise ValueError("Empty active cohort and no previous endpoint supplied")
        return previous_endpoint
    weights = [client_weights[k] * supervision_fractions[k] for k in active]
    return _weighted_average([local_endpoints[k] for k in active], weights)


def active_cohort_endpoint_average(
    local_endpoints: Mapping[Hashable, object],
    client_weights: Mapping[Hashable, float],
    supervision_fractions: Mapping[Hashable, float],
    previous_endpoint=None,
):
    """Focused ablation: same active cohort, server weights proportional to w_k only."""
    active = [k for k in local_endpoints if supervision_fractions.get(k, 0.0) > 0]
    if not active:
        if previous_endpoint is None:
            raise ValueError("Empty active cohort and no previous endpoint supplied")
        return previous_endpoint
    return _weighted_average([local_endpoints[k] for k in active], [client_weights[k] for k in active])

