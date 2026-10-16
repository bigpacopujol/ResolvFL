import csv
import json
import shutil
import pytest
from scripts.validate_repository import ROOT, validate


def test_registry_consistency():
    assert validate() == 76


def test_duplicate_record_is_rejected(tmp_path):
    shutil.copytree(ROOT / 'results/canonical', tmp_path / 'results/canonical')
    base = tmp_path / 'results/canonical/results'
    with base.with_suffix('.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    rows.append(rows[0].copy())
    with base.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    records = json.loads(base.with_suffix('.json').read_text())
    records.append(records[0].copy())
    base.with_suffix('.json').write_text(json.dumps(records))
    p = tmp_path / 'results/canonical/registry_status.json'
    status = json.loads(p.read_text())
    status['records'] += 1
    p.write_text(json.dumps(status))
    with pytest.raises(ValueError, match='Duplicate record'):
        validate(tmp_path)
