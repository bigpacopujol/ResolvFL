from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "results/canonical/manuscript_results.csv",
    ROOT / "docs/METRIC_CONTRACT.md",
    ROOT / "configs/cifar100.yaml",
    ROOT / "configs/inat_aves.yaml",
    ROOT / "configs/imagenet_3lf.yaml",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing required files: " + ", ".join(missing))

with (ROOT / "results/canonical/manuscript_results.csv").open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("Canonical manuscript export is empty")

required_columns = {
    "dataset", "experiment", "method_or_endpoint", "metric", "scope",
    "manuscript_eligible",
}
missing_columns = required_columns.difference(rows[0])
if missing_columns:
    raise SystemExit("Missing registry columns: " + ", ".join(sorted(missing_columns)))

metric_contract = {
    "leaf_accuracy",
    "leaf_accuracy_difference",
    "marginal_prefix_accuracy",
    "marginal_prefix_accuracy_difference",
    "conditional_head_accuracy",
    "raw_internal_node_prediction",
}

for row in rows:
    if row["manuscript_eligible"].lower() != "true":
        raise SystemExit("Canonical export contains a row outside the manuscript set")
    if row["metric"] not in metric_contract:
        raise SystemExit(f"Unknown metric in canonical export: {row['metric']}")

print(f"OK: {len(rows)} manuscript records validated against metric contract v1.0.0.")
