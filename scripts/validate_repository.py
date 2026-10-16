"""Validate result formats, record identities and metric definitions."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ('dataset', 'experiment', 'condition', 'method_or_endpoint', 'metric', 'scope', 'seed', 'run_index')
METRICS = {'leaf_accuracy', 'leaf_accuracy_difference', 'marginal_prefix_accuracy',
           'marginal_prefix_accuracy_difference', 'conditional_head_accuracy', 'raw_internal_node_prediction'}


def read_pair(base):
    with base.with_suffix('.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    normalized = []
    for row in rows:
        r = row.copy()
        for key in ('mean', 'sd', 'value'):
            r[key] = float(r[key]) if r[key] else None
        for key in ('seed', 'run_index'):
            r[key] = int(r[key]) if r[key] else None
        normalized.append(r)
    if normalized != json.loads(base.with_suffix('.json').read_text(encoding='utf-8')):
        raise ValueError(f'CSV/JSON mismatch: {base}')
    return rows


def validate(root=ROOT):
    accepted = read_pair(root / 'results/canonical/results')
    status = json.loads((root / 'results/canonical/registry_status.json').read_text())
    if len(accepted) != status['records']:
        raise ValueError('Registry counts disagree with registry_status.json')
    identities = set()
    for row in accepted:
        if row['metric'] not in METRICS:
            raise ValueError('Unknown metric in result registry')
        identity = tuple(row[k] for k in IDENTITY)
        if identity in identities:
            raise ValueError(f'Duplicate record: {identity}')
        identities.add(identity)
        if row['dataset'] == 'iNat-Aves' and row['experiment'] == 'paired_difference' and row['value']:
            if row['seed'] or not row['run_index']:
                raise ValueError('iNat presentation indices must not be labelled as seeds')
    return len(accepted)


if __name__ == '__main__':
    count = validate()
    print(f'OK: {count} records; CSV/JSON consistent.')

