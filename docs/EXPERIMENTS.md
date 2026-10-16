# Experimental protocol

The [training notebook guide](RECOVERED_EXPERIMENTS.md) lists the sources for each dataset. Shared protocol settings are:

- 10 clients;
- 150 communication rounds;
- one local epoch per round;
- full participation;
- final-round evaluation;
- SGD with momentum 0.9 for CIFAR-100 and iNat-Aves, and without momentum for ImageNet-3L/F;
- optimizer recreation at the beginning of each communication round;
- weight decay 0 for the audited node-head dynamics.

Dataset-specific settings:

| Dataset | Representation | Batch | LR | Runs |
|---|---|---:|---:|---|
| CIFAR-100 | trainable ConvNet | 128 | 0.05 | 1, 2, 3 |
| iNat-Aves | frozen ImageNet RN50 features + trainable projection | 64 | 0.01 | 3 paired runs |
| ImageNet-3L/F | fixed DINO ResNet-50 | 512 | 0.05 | 1, 2, 3 |

Default federations use the two-stage double-Dirichlet construction with `alpha_cls = alpha_dep = 0.5`.

For the three-run comparisons, the repository reports descriptive means, sample standard deviations, and paired differences.

## Controlled cohort-overlap sweep

For the CIFAR controlled sweep, all 10 clients supervise the root decision. The deep cohort is a subset of size 10, 8, 5, 2, or 1. Its Jaccard overlap with the root cohort is therefore 1.00, 0.80, 0.50, 0.20, or 0.10, respectively. These are controlled settings, not measured overlap values for the default Dirichlet federations.
