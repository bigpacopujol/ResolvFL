# Release candidate 0.2.0-rc4 — 2026-10-15

- Rename the method to ResolvFL, the Python package to `resolvfl`, and the aggregation entry point to `resolvfl_endpoint_average`.
- Align the accepted registry with the consolidated MASTER: 76 accepted records and four separately quarantined iNat summaries; preserve all accepted numerical values.
- Separate iNat presentation indices from random seeds.
- Generate all figures from the accepted registry. Show available paired observations in the optional comparison figure.
- Compute empirical supervision counts bottom-up in linear time and locate maximal uncovered roots with one top-down traversal.
- Validate registry CSV/JSON agreement and prevent quarantined summaries from re-entering accepted exports.
- Document the missing end-to-end training sources and execution evidence explicitly.

# Release candidate 0.2.0-rc2 — 2026-09-16

This release candidate aligns the repository with the current Pattern Recognition manuscript.

Highlights:

- exact marginalized path supervision for mixed-granularity labels;
- tree-factorized leaf and prefix prediction;
- empirical node-level coverage auditing;
- node-specific ResolvFL aggregation and the active-cohort weighting ablation;
- double-Dirichlet federation construction and controlled cohort-overlap utilities;
- canonical CIFAR-100, iNat-Aves, and ImageNet-3L/F manuscript results;
- scripts for the current manuscript tables and figures;
- unit tests for the core taxonomy, loss, coverage, aggregation, partition, and metric components;
- dataset-specific experimental configurations matching the current manuscript protocol.

