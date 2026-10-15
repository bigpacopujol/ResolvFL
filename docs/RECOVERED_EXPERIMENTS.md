# Recovered experiment sources — 2026-10-15

The author supplied the original Colab notebooks, Python scripts and saved run archives. These sources are available under `experiments/recovered/`; recovered JSON outputs are under `results/recovered/`. Their historical FedGrain names identify the original runs of the method now named ResolvFL. Executable identifiers, filenames and paths are retained because checkpoint names and resume paths depend on them. Notebook prose and selected comments have been edited without changing executable tokens.

## Which source to use

| Source | Role |
|---|---|
| `FedGrain_CIFAR100_seed_paired.ipynb` | CIFAR training and seed-paired main comparison; section 14 identifies partition seeds 10001–10003 and training seeds 20001–20003. |
| `FedGrain_CIFAR100.ipynb` | Earlier CIFAR source, retained for provenance. |
| `FedGrain_iNat_Aves.ipynb` | iNat training, feature loading, main runs and paired coverage intervention. |
| `FedGrain_ImageNet3LF_audited_violation.ipynb` | ImageNet exact path-loss implementation and v4 full-run workflow. Main runs are optional and disabled by default in this saved copy. |
| `FedGrain_ImageNet3LF.ipynb`, `train_imagenet3lf.py` | Earlier ImageNet node-balanced-loss sources. Do not use these to reproduce the accepted main table. |
| `extract_dino_features.py` | DINO feature-extraction source; input manifests and label-column agreement still require verification. |

Open notebooks in Colab and inspect their data/checkpoint paths before running. They are preserved sources, not an automatic top-to-bottom reproduction recipe: some cells resume, skip completed runs, or remove an earlier checkpoint. None was executed during this audit. All five notebooks were supplied without saved cell outputs.

## Numerical evidence recovered

The saved CIFAR main JSON, iNat summary plus multiseed JSON, and ImageNet `v4_exact_pathloss_violation_full/imagenet3lf_main_results_audited.json` reproduce all nine accepted main leaf-accuracy means and sample SDs at manuscript precision. This is numerical provenance, not a rerun of training.

CIFAR raw overlap observations and the no-extra-mass ablation are recovered. iNat source identifies main seed labels 1, 2, 3 (seed 1 in `results_summary.json`, seeds 2 and 3 in `multiseed_main.json`) and intervention seeds 10, 20, 30. Keep canonical presentation indices unchanged until the full provenance mapping is accepted. ImageNet v4 records the selected subordinate node 24 and its five leaf IDs 153–157.

The accepted registry remains unchanged. Raw evidence includes obsolete metrics and experiments; filenames containing `canonical` do not grant manuscript eligibility.

## Static implementation audit

- CIFAR and iNat sum cross-entropies over proper-ancestor decisions and divide by the full minibatch size. Both recreate SGD for each client in each round, with momentum 0.9 and default zero weight decay.
- CIFAR and iNat aggregate shared backbone/projection states across all clients by sample count. Node-head weights are proportional to client sample count times local supervision fraction. An empty cohort leaves the first local head state; equality with the broadcast state relies on the head receiving no updates.
- ImageNet's audited notebook uses sum-reduced path terms divided by full minibatch size, fixed input features, supervision-count weights for active heads, and explicit broadcast retention for an empty cohort.
- **Protocol discrepancy:** ImageNet's audited tree and flat loops construct `SGD(parameters, lr=LR)` without momentum, hence momentum 0. This differs from the manuscript's general momentum-0.9 description. The recovered source must not be silently modified to 0.9 and represented as the executed code. Reconcile the manuscript protocol with this evidence.
- Runtime versions and exact resume histories are not established by static source inspection. No claim of full execution equivalence is made.

## Metrics and remaining dependencies

The intervention code in these recovered notebooks computes ancestor-of-decoded-leaf metrics. Those outputs are not marginal prefix accuracy. The later evaluation scripts/outputs supporting the accepted marginal-prefix values have not been recovered in this batch. Keep them separate; never replace the accepted marginal results with the older JSON fields.

Petroica remains **five species**, in the accepted experimental setting. A historical summary reports four; that stale field is not promoted or used to revise the accepted setting.

Missing inputs include `fedgrain_inat/aves_features.npz` and its preparation pipeline, and ImageNet `train_labels.npy`, `val_labels.npy`, `train_features.npy`, `val_features.npy`. The supplied `features dino.zip` contains only metadata/progress JSON. Its metadata lists two label columns, while the audited training loader requires at least three; the actual final arrays must resolve this lineage issue. Large feature arrays need not be committed to GitHub.

Model checkpoint archives were supplied but are not committed here. Some local ZIP copies became truncated on a subsequent workspace read; `source_manifest.json` records this, and only complete JSON entries with verified ZIP CRCs were extracted from those copies. Model binaries were not deserialized or executed. Earlier complete archive inspection and the extracted sources do not substitute for a complete checkpoint audit.

## Checks

Run `python scripts/check_recovered_results.py` to compare recovered main results with the accepted registry. This check reads saved outputs only and does not train models. Original attachment hashes and current edited-file SHA-256 hashes, together with unchanged JSON evidence hashes, are in `results/recovered/source_manifest.json`.
