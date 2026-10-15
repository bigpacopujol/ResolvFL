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

The registry records the provenance level available for each result. These annotations do not replace original run logs or checkpoints. The six iNat main paired observations retain presentation indices in `run_index`. Recovered source/JSON now identifies main seed labels 1–3; the canonical mapping has not been changed.

## Current release scope

Core computations, the accepted artifact registry, recovered Colab/Python training sources and saved JSON outputs are available. [RECOVERED_EXPERIMENTS.md](RECOVERED_EXPERIMENTS.md) identifies valid versus historical sources and records the static audit. Data/feature inputs, later marginal-prefix evaluation provenance and the ImageNet momentum discrepancy remain open. The accepted numerical registry is unchanged; raw recovered outputs are not automatically eligible for manuscript exports.

## Coverage implementation

Observed-label counts are accumulated bottom-up. Each node and edge is visited a constant number of times, so per-client counting takes O(|V| + |E|) time with dictionary lookups and O(|V|) auxiliary storage. A label at a node contributes only to its proper ancestors. Maximal uncovered roots are found in a separate top-down traversal.
