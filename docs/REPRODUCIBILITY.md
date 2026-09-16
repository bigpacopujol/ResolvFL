# Reproducibility

The repository is organized around two reproducibility layers: the **core FedGrain computations** and the **manuscript-facing artifact pipeline**.

## Core computations

The package provides tested implementations of:

- rooted-taxonomy traversal and subtree operations;
- exact marginalized path negative log-likelihood;
- empirical node-level supervision counts and coverage flags;
- FedGrain node-specific active-cohort aggregation;
- the active-cohort-only weighting ablation;
- leaf, marginal-prefix, conditional-head, and Flat raw-internal metrics;
- double-Dirichlet allocation and the controlled cohort-overlap construction.

Run the unit tests with:

```bash
pytest -q
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

The canonical result registry also records provenance at the level available for each experiment. This keeps manuscript values, paired comparisons, and intervention endpoints traceable to their audited source artifacts.

## Current release scope

This release concentrates on the audited FedGrain components and on exact regeneration of manuscript tables and figures from the canonical result registry. The dataset-specific training entry points will be added when the final public training workflow is consolidated from the audited experiment sources.
