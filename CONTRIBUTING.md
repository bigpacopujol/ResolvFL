# Contributing

Contributions should preserve the repository's scientific contracts:

- metric semantics are defined in `docs/METRIC_CONTRACT.md`;
- experiment settings are defined in `docs/EXPERIMENTS.md` and `configs/`;
- manuscript-facing numerical results are maintained in `results/canonical/manuscript_results.csv`;
- changes to the core loss, coverage, aggregation, partition, or metric code should include corresponding tests.

Before submitting a change, run:

```bash
pytest -q
python scripts/validate_repository.py
```

