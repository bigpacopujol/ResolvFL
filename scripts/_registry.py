"""Load and select experimental result records."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]


def load_rows():
    with (ROOT / 'results/canonical/results.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Expected a nonempty result registry')
    return rows


def select(rows, **criteria):
    return [r for r in rows if all(r[k] == v for k, v in criteria.items())]


def one(rows, **criteria):
    matches = select(rows, **criteria)
    if len(matches) != 1:
        raise ValueError(f'Expected one record for {criteria}; found {len(matches)}')
    return matches[0]


def save_figure(fig, name):
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight')

