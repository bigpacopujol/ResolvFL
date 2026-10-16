import csv
import json
import shutil
import pytest
from scripts.validate_repository import ROOT, validate


def test_registry_matches_accepted_master_scope():
    assert validate() == (76, 4)


def test_quarantined_summary_cannot_be_reintroduced(tmp_path):
    shutil.copytree(ROOT / 'results', tmp_path / 'results')
    base = tmp_path / 'results/canonical/manuscript_results'
    with base.with_suffix('.csv').open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    with (tmp_path / 'results/quarantine/unpromoted_results.csv').open(newline='') as f:
        rejected = next(csv.DictReader(f))
    rejected['manuscript_eligible'] = 'true'; rows.append(rejected)
    with base.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    j = json.loads(base.with_suffix('.json').read_text())
    for key in ('mean', 'sd', 'value'):
        rejected[key] = float(rejected[key]) if rejected[key] else None
    for key in ('seed', 'run_index'):
        rejected[key] = int(rejected[key]) if rejected[key] else None
    j.append(rejected); base.with_suffix('.json').write_text(json.dumps(j))
    p = tmp_path / 'results/canonical/registry_status.json'
    status = json.loads(p.read_text()); status['accepted_records'] += 1; p.write_text(json.dumps(status))
    with pytest.raises(ValueError, match='Quarantined summary reintroduced'):
        validate(tmp_path)

