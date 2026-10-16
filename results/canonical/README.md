# Experimental results

`results.csv` contains the summary results for CIFAR-100, iNat-Aves and ImageNet-3L/F. `results.json` provides the same records in JSON format.

Each record identifies the dataset, experiment, method, metric and evaluation subset, together with its mean and sample standard deviation or individual run value. Accuracies are percentages; paired differences are percentage points.

`seed` identifies a recorded random seed. `run_index` denotes presentation order when the record does not specify a seed. Empty fields indicate values not supplied for that record.

The 76 records cover the main comparisons, paired differences and coverage experiments. Table and figure scripts read this registry; per-run outputs are in `../runs/`. Metric definitions are in `../../docs/METRIC_CONTRACT.md`.
