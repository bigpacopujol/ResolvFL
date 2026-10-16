from pathlib import Path
from resolvfl.provenance import build_manifest
ROOT=Path(__file__).resolve().parents[1]
rows=build_manifest(ROOT)
(ROOT/"MANIFEST.sha256").write_text("".join(f"{h}  {p}\n" for h,p in rows),encoding="utf-8")
print(f"Wrote {len(rows)} entries")


