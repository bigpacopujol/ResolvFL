# Experimental protocol

The default federated protocol uses:

- 10 clients;
- 150 communication rounds;
- one local epoch per round;
- full participation;
- final-round evaluation;
- SGD with momentum 0.9;
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

For the CIFAR controlled sweep, all clients supervise the root decision and a fixed subset supervises the deeper leaf-level decisions. With 10 clients, the nominal settings `1.00, 0.75, 0.50, 0.25, 0.10` correspond to realized active-cohort overlaps `1.00, 0.80, 0.50, 0.20, 0.10`.
