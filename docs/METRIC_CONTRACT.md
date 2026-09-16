# Metric contract v1.0.0

FedGrain uses four metric families. Their decoder and evaluation scope are fixed for all manuscript-facing results.

## Leaf accuracy

For Tree/FedGrain, the global posterior of a leaf is the product of conditional probabilities along its root-to-leaf path. Prediction is the leaf with maximum global posterior.

For Flat, the output distribution is restricted to leaf nodes, renormalized over those leaves, and decoded by leaf argmax.

Target-subtree and outside-subtree accuracies are evaluation scopes of the same leaf-accuracy definition.

## Marginal prefix accuracy

At a declared taxonomy level or antichain, each candidate node is scored by its marginal subtree probability, equivalently the path product truncated at that node. Prediction is the candidate with maximum marginal probability.

This metric evaluates the resolved prefix decision directly at the requested taxonomic level.

## Conditional-head accuracy

For a selected internal node, the true parent is supplied and the local child head is evaluated directly. This is a node-level diagnostic of the corresponding conditional decision.

## Raw internal-node prediction

For Flat models, this diagnostic reports the fraction of examples whose unrestricted argmax over all taxonomy-node outputs is an internal node.

## Implementation

The reference implementations are in `src/fedgrain/metrics.py`, and unit tests are in `tests/test_metrics.py`.
