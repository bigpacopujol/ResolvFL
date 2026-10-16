# Dataset setup

ResolvFL experiments use three hierarchical recognition benchmarks.

- **CIFAR-100** — native superclass-to-leaf hierarchy with 20 superclasses and 100 fine classes.
- **iNat-Aves** — Aves subset derived from iNaturalist 2021, using the order–family–genus–species hierarchy. Experiments use frozen ImageNet-pretrained ResNet-50 features followed by a trainable shared projection and node heads.
- **ImageNet-3L/F** — ImageNet-derived basic–subordinate–leaf hierarchy. Experiments use fully fixed DINO ResNet-50 features feeding the node heads directly.

Obtain the datasets from their official distributions under the corresponding access and licensing terms. Dataset-specific experiment settings are stored in `configs/` and summarized in `docs/EXPERIMENTS.md`.


