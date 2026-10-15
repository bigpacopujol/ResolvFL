import matplotlib.pyplot as plt
import numpy as np
from _registry import load_rows, one, save_figure

rows = load_rows()
targets = [
    ('CIFAR-100', 'vehicles_1', 'target leaves', 'CIFAR\nvehicles_1'),
    ('CIFAR-100', 'small_mammals', 'target leaves', 'CIFAR\nsmall_mammals'),
    ('CIFAR-100', 'flowers', 'target leaves', 'CIFAR\nflowers'),
    ('iNat-Aves', 'Petroica', 'target species', 'iNat-Aves\nPetroica'),
    ('ImageNet-3L/F', 'median-support five-leaf subordinate', 'target leaves', 'ImageNet\n5-leaf subtree'),
]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
x = np.arange(len(targets)); width = 0.38
for endpoint, offset, hatch in [('Reference', -width/2, None), ('Coverage removed', width/2, '///')]:
    values = [float(one(rows, dataset=ds, experiment='coverage_removal', condition=condition,
                        metric='leaf_accuracy', scope=scope, method_or_endpoint=endpoint)['mean'])
              for ds, condition, scope, _ in targets]
    ax.bar(x + offset, values, width, label=endpoint, hatch=hatch)
ax.set_ylabel('Target leaf accuracy (%)')
ax.set_xticks(x, [t[3] for t in targets])
ax.set_ylim(0, 72); ax.grid(axis='y', alpha=.35)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, ncol=2, loc='upper left')
fig.tight_layout()
save_figure(fig, 'fig_target_subtree_violation')
