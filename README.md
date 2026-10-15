# ResolvFL

Core code and manuscript artifacts for **Identifiable Fine-Grained Recognition from Federated Mixed-Granularity Supervision**.

ResolvFL addresses fine-grained recognition when federated clients annotate examples at different depths of a shared taxonomy. It combines four components:

1. **tree-factorized prediction**, with one conditional decision at each internal taxonomy node;
2. **exact marginalized path supervision**, so every observed label trains the decisions on its root-to-label path;
3. **empirical coverage auditing**, which identifies the internal decisions represented by the observed federation;
4. **node-specific federated aggregation**, using the clients that supervise each decision.

The repository contains the core implementation, protocol configurations, metric definitions, accepted manuscript results, and scripts that regenerate tables and figures from that registry. Recovered Colab training sources and saved JSON run outputs are now included. See [the recovered-experiment guide](docs/RECOVERED_EXPERIMENTS.md) for source selection, verified numerical matches and remaining dependencies. End-to-end execution equivalence is not yet certified.

## Quick start

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Minimal example

### 1. Define a taxonomy

```python
from resolvfl.taxonomy import Taxonomy

T = Taxonomy(
    root="root",
    children={
        "root": ["animal", "vehicle"],
        "animal": ["cat", "dog"],
        "vehicle": ["car", "truck"],
    },
)
```

### 2. Train from a mixed-granularity label

```python
from resolvfl.loss import exact_path_nll

loss = exact_path_nll(
    logits_by_node=logits_by_node,
    observed_node="animal",
    taxonomy=T,
)
```

For the coarse label `animal`, the loss evaluates the observed root-to-`animal` path. A leaf label such as `cat` extends the same construction through the corresponding fine-grained decision.

### 3. Audit empirical supervision coverage

```python
from resolvfl.coverage import empirical_coverage_flags

flags = empirical_coverage_flags(client_histograms, T)
```

`flags[u]` indicates whether the observed federation contains examples that supervise internal decision `u`.

### 4. Aggregate a node head across its active cohort

```python
from resolvfl.aggregation import resolvfl_endpoint_average

updated_head = resolvfl_endpoint_average(
    local_endpoints=local_heads,
    client_weights=client_weights,
    supervision_fractions=supervision_fractions,
    previous_endpoint=global_head,
)
```

## Reproduce manuscript artifacts

The manuscript-facing tables and figures are generated from `results/canonical/manuscript_results.csv`:

```bash
python scripts/validate_repository.py
python scripts/export_manuscript_tables.py
python scripts/generate_fig2a.py
python scripts/generate_fig2b.py
python scripts/generate_fig3.py
```

Generated tables are written under `results/derived/`; figure scripts write the corresponding PDF/PNG files under `figures/`.

## Main experimental settings

| Dataset | Taxonomy | Representation | Batch | LR | Runs |
|---|---|---|---:|---:|---|
| CIFAR-100 | superclass–leaf | trainable ConvNet | 128 | 0.05 | 1, 2, 3 |
| iNat-Aves | order–family–genus–species | frozen ImageNet ResNet-50 features + trainable projection | 64 | 0.01 | 3 paired runs |
| ImageNet-3L/F | basic–subordinate–leaf | fixed DINO ResNet-50 | 512 | 0.05 | 1, 2, 3 |

All default experiments use 10 clients, 150 communication rounds, one local epoch, full participation, final-round evaluation, and the two-stage double-Dirichlet construction with `alpha_cls = alpha_dep = 0.5`.

### Main leaf-accuracy results (%)

| Dataset | Flat | Tree | ResolvFL |
|---|---:|---:|---:|
| CIFAR-100 | 37.74 ± 2.60 | 37.86 ± 1.52 | 38.08 ± 2.12 |
| iNat-Aves | 29.10 ± 0.77 | 38.25 ± 2.31 | 40.45 ± 2.34 |
| ImageNet-3L/F | 52.68 ± 7.77 | 59.52 ± 3.71 | 59.38 ± 2.67 |

The paired comparisons and coverage-removal interventions used in the paper are available in `results/canonical/` and `results/derived/`. Four unpromoted iNat summaries are retained separately in `results/quarantine/` and are excluded from manuscript exports. The optional paired-comparison figure shows individual paired observations where available; it does not infer new summaries.

## Metrics

The metric definitions used throughout the repository are fixed in `docs/METRIC_CONTRACT.md`:

- leaf accuracy;
- marginal prefix accuracy;
- conditional-head accuracy;
- Flat raw internal-node prediction.

## Repository layout

```text
src/resolvfl/       core ResolvFL implementation
configs/            dataset and protocol configurations
tests/              unit tests for loss, taxonomy, coverage, aggregation and metrics
results/canonical/  manuscript-facing result registry
results/derived/    generated manuscript tables
scripts/            validation and figure/table generation
docs/               dataset, experiment and metric documentation
figures/             regenerated manuscript figures
```

## Reproducibility

`docs/REPRODUCIBILITY.md` describes the artifact workflow and the provenance recorded with each manuscript-facing result. `docs/EXPERIMENTS.md` contains the experimental protocol and dataset-specific settings.

## Name migration

The method and Python package are now **ResolvFL** and `resolvfl`. Update imports to `resolvfl` and use `resolvfl_endpoint_average`. The repository is now `bigpacopujol/ResolvFL`. Original recovered experiment files retain historical names to preserve checkpoint references. Historical result values are unchanged.

## Citation

Citation metadata are provided in `CITATION.cff`. The associated manuscript is:

**Identifiable Fine-Grained Recognition from Federated Mixed-Granularity Supervision**.

