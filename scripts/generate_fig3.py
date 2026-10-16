"""Optional comparison figure: paired observations where available, otherwise mean/SD."""
import matplotlib.pyplot as plt
from _registry import load_rows, select, one, save_figure

rows = load_rows()
fig, ax = plt.subplots(figsize=(8.4, 4.8))
labels = []; used = set()
for dataset in ('CIFAR-100', 'iNat-Aves', 'ImageNet-3L/F'):
    for contrast in ('Tree-Flat', 'ResolvFL-Tree'):
        y = len(labels)
        labels.append(f"{dataset}: {contrast.replace('-', ' - ')}")
        records = select(rows, dataset=dataset, experiment='paired_difference', condition='default',
                         method_or_endpoint=contrast, metric='leaf_accuracy_difference', scope='all leaves')
        paired = [r for r in records if r['value'] != '']
        if paired:
            label = 'Paired observations'
            ax.scatter([float(r['value']) for r in paired], [y] * len(paired), color='C0',
                       marker='o', label=label if label not in used else None)
        else:
            summary = one(records, value='')
            label = 'Mean ± sample SD'
            ax.errorbar(float(summary['mean']), y, xerr=float(summary['sd']), fmt='s', color='C1',
                        capsize=3, label=label if label not in used else None)
        used.add(label)
ax.axvline(0, linestyle='--', linewidth=1)
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel('Leaf-accuracy difference (percentage points)')
ax.invert_yaxis(); ax.grid(axis='x', alpha=.35)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2)
fig.tight_layout()
save_figure(fig, 'fig_paired_differences')

