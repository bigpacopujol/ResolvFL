from resolvfl.taxonomy import Taxonomy
from resolvfl.metrics import (
    METRIC_REGISTRY, tree_leaf_posterior, flat_leaf_posterior,
    marginal_prefix_prediction, conditional_head_prediction,
    raw_internal_node_prediction,
)


def toy():
    return Taxonomy("r", {"r": ["A", "B"], "A": ["a1", "a2"], "B": ["b1", "b2"]})


def test_metric_registry_contract():
    assert set(METRIC_REGISTRY) == {
        "leaf_accuracy",
        "marginal_prefix_accuracy",
        "conditional_head_accuracy",
        "raw_internal_node_prediction",
    }


def test_tree_and_marginal_decoders():
    t = toy()
    probs = {"r": [0.7, 0.3], "A": [0.2, 0.8], "B": [0.9, 0.1]}
    leaves = tree_leaf_posterior(probs, t)
    assert max(leaves, key=leaves.get) == "a2"
    assert marginal_prefix_prediction(["A", "B"], probs, t) == "A"
    assert conditional_head_prediction("A", probs, t) == "a2"


def test_flat_leaf_renormalization_and_internal_diag():
    t = toy()
    flat = {"A": 0.5, "B": 0.1, "a1": 0.05, "a2": 0.15, "b1": 0.12, "b2": 0.08}
    leaf = flat_leaf_posterior(flat, t)
    assert abs(sum(leaf.values()) - 1.0) < 1e-12
    assert raw_internal_node_prediction(flat, t) is True

