"""Validate artifact consistency and explicit exclusions; does not certify training runs."""
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
    accepted = read_pair(root / 'results/canonical/manuscript_results')
    excluded = read_pair(root / 'results/archive/additional_summaries')
    status = json.loads((root / 'results/canonical/registry_status.json').read_text())
    if len(accepted) != status['accepted_records'] or len(excluded) != status['archived_records']:
        raise ValueError('Registry counts disagree with registry_status.json')
    identities = set()
    for row in accepted:
        if row['manuscript_eligible'] != 'true' or row['metric'] not in METRICS:
            raise ValueError('Invalid eligibility or metric in accepted registry')
        identity = tuple(row[k] for k in IDENTITY)
        if identity in identities:
            raise ValueError(f'Duplicate accepted record: {identity}')
        identities.add(identity)
        if row['dataset'] == 'iNat-Aves' and row['experiment'] == 'paired_difference' and row['value']:
            if row['seed'] or not row['run_index']:
                raise ValueError('iNat presentation indices must not be labelled as seeds')
    for row in excluded:
        if row['manuscript_eligible'] != 'false':
            raise ValueError('Archived record marked eligible')
        if tuple(row[k] for k in IDENTITY) in identities:
            raise ValueError('Archived summary reintroduced in accepted registry')
    return len(accepted), len(excluded)


if __name__ == '__main__':
    accepted, excluded = validate()
    print(f'OK: {accepted} accepted records; {excluded} archived; CSV/JSON consistent. This validates artifacts, not executed training.')

