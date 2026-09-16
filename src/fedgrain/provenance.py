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
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
        rows.append((sha256_file(path), path.relative_to(root).as_posix()))
    return rows
