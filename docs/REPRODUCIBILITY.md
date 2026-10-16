# Reproducibility

The repository is organized around two reproducibility layers: the **core ResolvFL computations** and the **table and figure workflow**.

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

## Table and figure generation

`results/canonical/results.csv` is the source used by the table and figure scripts. Each record includes its dataset, experiment, metric, evaluation scope, summary value, and provenance level.

Validate the registry:

```bash
python scripts/validate_repository.py
```

Generate tables and figures:

```bash
python scripts/export_tables.py
python scripts/generate_fig2a.py
python scripts/generate_fig2b.py
python scripts/generate_fig3.py
```

## Experimental configurations

Dataset-specific configurations are stored in `configs/` and summarized in `docs/EXPERIMENTS.md`. The default protocol uses 10 clients, 150 communication rounds, one local epoch per round, full participation, and final-round evaluation.

Run `python scripts/check_results.py` to compare the saved per-run main results with the summary registry. The [training guide](TRAINING.md) describes the notebooks, required inputs and checkpoint options.

## Coverage implementation

Observed-label counts are accumulated bottom-up. Each node and edge is visited a constant number of times, so per-client counting takes O(|V| + |E|) time with dictionary lookups and O(|V|) auxiliary storage. A label at a node contributes only to its proper ancestors. Maximal uncovered roots are found in a separate top-down traversal.

