# Training notebooks and saved results

## Choose a notebook

| Dataset or task | Source | Purpose |
|---|---|---|
| CIFAR-100 | `experiments/ResolvFL_CIFAR100_seed_paired.ipynb` | Training and paired main comparison, with partition seeds 10001–10003 and training seeds 20001–20003. |
| iNat-Aves | `experiments/ResolvFL_iNat_Aves.ipynb` | Feature loading, training, main comparison and Petroica coverage intervention. |
| ImageNet-3L/F | `experiments/ResolvFL_ImageNet3LF_audited_violation.ipynb` | Exact path-loss training and coverage intervention. Enable the optional main-comparison section to run the main experiments. |
| Feature extraction | `experiments/extract_dino_features.py` | Extract frozen DINO features from the configured image manifests. |

`ResolvFL_CIFAR100.ipynb` contains the initial CIFAR experiments. `ResolvFL_ImageNet3LF.ipynb` and `train_imagenet3lf.py` implement the earlier node-balanced loss; use the exact-path-loss notebook above for the main comparison.

## Run in Colab

1. Open the selected notebook and choose a GPU runtime.
2. Set its dataset, feature and checkpoint paths to your own locations before running cells.
3. Run the setup and model-definition cells, followed by the required experiment sections.
4. Check the checkpoint options: some sections resume existing runs, skip completed configurations or reset a checkpoint for a controlled experiment.

The notebooks use `resolvfl` in generated module names, checkpoint paths and result keys. Existing files stored under a different name must be renamed consistently or supplied through the notebook's path settings. No files in the user's Drive are changed by a repository update.

## Inputs

- CIFAR-100: the dataset loaded by the CIFAR notebook.
- iNat-Aves: `aves_features.npz`, containing the feature and taxonomy arrays expected by the notebook loader. Features are frozen ImageNet-pretrained ResNet-50 representations; the shared projection and heads are trained.
- ImageNet-3L/F: `train_features.npy`, `val_features.npy`, `train_labels.npy` and `val_labels.npy`. Labels must provide the basic, subordinate and leaf columns expected by the training loader. The feature-extraction script uses image manifests configured for the local dataset.

Datasets, feature arrays and model checkpoints are supplied separately from this repository.

## Protocol and metrics

See [EXPERIMENTS.md](EXPERIMENTS.md) for settings and [METRIC_CONTRACT.md](METRIC_CONTRACT.md) for metric definitions. CIFAR-100 and iNat-Aves use SGD momentum 0.9; ImageNet-3L/F uses SGD without momentum.

Some notebook intervention summaries evaluate the ancestor of the decoded leaf. This diagnostic differs from marginal-prefix accuracy, which predicts using subtree marginal probabilities. Select the metric required by the experiment when comparing outputs.

## Saved results

`results/runs/` contains per-run JSON outputs. The main comparison uses:

- CIFAR-100: `resolvfl_cifar100/table1_multiseed_partitioned_v2.json`;
- iNat-Aves: `resolvfl_inat_aves/results_summary.json` (seed 1) and `multiseed_main.json` (seeds 2 and 3);
- ImageNet-3L/F: `imagenet3lf_ckpt/v4_exact_pathloss_violation_full/imagenet3lf_main_results_audited.json`.

Run `python scripts/check_results.py` to compare the nine main leaf-accuracy means and sample standard deviations with `results/canonical/results.csv`. This command reads saved results. Table and figure generation commands are listed in the README.
