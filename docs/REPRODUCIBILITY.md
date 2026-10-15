# Reproducibility

The repository is organized around two reproducibility layers: the **core ResolvFL computations** and the **manuscript-facing artifact pipeline**.

## Core computations

The package provides tested implementations of:

- rooted-taxonomy traversal and subtree operations;
- exact marginalized path negative log-likelihood;
- empirical node-level supervision counts and coverage flags;
- ResolvFL node-specific active-cohort aggregation;
- the active-cohort-only weighting ablation;
- leaf, marginal-prefix, conditional-head, and Flat raw-internal metrics;
- double-Dirichlet allocation and the controlled cohort-overlap construction.

Run the unit tests with:

```bash
python -m pytest -q
```

## Manuscript artifact pipeline

`results/canonical/manuscript_results.csv` is the source used by the table and figure scripts. Each record includes its dataset, experiment, metric, evaluation scope, summary value, and provenance level.

Validate the registry:

```bash
python scripts/validate_repository.py
```

Regenerate manuscript artifacts:

```bash
python scripts/export_manuscript_tables.py
python scripts/generate_fig2a.py
python scripts/generate_fig2b.py
python scripts/generate_fig3.py
```

## Experimental configurations

Dataset-specific configurations are stored in `configs/` and summarized in `docs/EXPERIMENTS.md`. The default protocol uses 10 clients, 150 communication rounds, one local epoch per round, full participation, and final-round evaluation.

The registry records the provenance level available for each result. These annotations do not replace original run logs or checkpoints. The six iNat main paired observations have presentation indices in `run_index`; their original seeds and run identifiers remain unknown.

## Current release scope

This release provides core computations and regeneration of manuscript artifacts from the accepted result registry. It does not contain the dataset-specific model definitions, feature extraction pipeline, training loop, dataset taxonomy assets, or original run logs/checkpoints needed for end-to-end experiment reproduction.

Protocol configurations document the intended manuscript settings. They cannot establish which settings were executed. In particular, minibatch loss reduction, optimizer recreation, masking, and shared-parameter aggregation still require comparison with the original training source. The iNat main run IDs and the taxonomy membership/selection of the intervention subtrees also remain open.

The registry contains 76 accepted records. Four historical iNat summaries are kept in `results/quarantine/` pending provenance verification and an explicit promotion decision. The validator checks exclusion, CSV/JSON consistency, uniqueness and metric names; passing it does not certify experimental reproducibility.

## Coverage implementation

Observed-label counts are accumulated bottom-up. Each node and edge is visited a constant number of times, so per-client counting takes O(|V| + |E|) time with dictionary lookups and O(|V|) auxiliary storage. A label at a node contributes only to its proper ancestors. Maximal uncovered roots are found in a separate top-down traversal.
