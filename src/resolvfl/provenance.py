from __future__ import annotations
import hashlib
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root):
    root = Path(root)
    rows = []
    ignored = {'.git', '__pycache__', '.pytest_cache', '.venv', 'build', 'dist'}
    files = (p for p in root.rglob('*') if p.is_file()
             and p.name != 'MANIFEST.sha256'
             and not any(part in ignored or part.endswith('.egg-info') for part in p.relative_to(root).parts)
             and p.suffix not in {'.pyc', '.pyo'})
    for path in sorted(files):
        rows.append((sha256_file(path), path.relative_to(root).as_posix()))
    return rows

