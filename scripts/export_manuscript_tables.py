from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "results/canonical/manuscript_results.csv").open(encoding="utf-8")))
out = ROOT / "results/derived"
out.mkdir(exist_ok=True)

# Table 4: main leaf accuracy + Flat raw internal diagnostic.
datasets = ["CIFAR-100", "iNat-Aves", "ImageNet-3L/F"]
methods = ["Flat", "Tree", "FedGrain"]
lines = ["dataset,flat_leaf,tree_leaf,fedgrain_leaf,flat_raw_internal"]
for ds in datasets:
    vals = {}
    for r in rows:
        if r["dataset"] != ds or r["experiment"] != "main_comparison" or r["condition"] != "default":
            continue
        if r["metric"] == "leaf_accuracy" and r["method_or_endpoint"] in methods:
            vals[r["method_or_endpoint"]] = f'{float(r["mean"]):.2f} +/- {float(r["sd"]):.2f}'
        if r["metric"] == "raw_internal_node_prediction" and r["method_or_endpoint"] == "Flat":
            vals["raw"] = f'{float(r["mean"]):.2f} +/- {float(r["sd"]):.2f}'
    lines.append(",".join([ds, vals["Flat"], vals["Tree"], vals["FedGrain"], vals["raw"]]))
(out / "table_main_comparison.csv").write_text("\n".join(lines)+"\n", encoding="utf-8")

# Table 3: deep coverage-removal endpoints.
keep = []
for r in rows:
    if r["experiment"] == "coverage_removal" and r["dataset"] in {"iNat-Aves","ImageNet-3L/F"}:
        keep.append(r)
with (out / "table_deep_coverage_removal.csv").open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=keep[0].keys()); w.writeheader(); w.writerows(keep)
print(out)
