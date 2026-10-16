"""Check saved main-table outputs and experiment file hashes."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/runs'

def read(path):
    return json.loads((RAW / path).read_text())

def main():
    for item in read('source_manifest.json')['files']:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    with (ROOT / 'results/canonical/manuscript_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    cifar = read('resolvfl_cifar100/table1_multiseed_partitioned_v2.json')['seeds']
    inat = read('resolvfl_inat_aves/multiseed_main.json')
    first = read('resolvfl_inat_aves/results_summary.json')
    imagenet = read('imagenet3lf_ckpt/v4_exact_pathloss_violation_full/imagenet3lf_main_results_audited.json')
    values = {}
    for method, key in [('Flat', 'flat'), ('Tree', 'tree'), ('ResolvFL', 'resolvfl')]:
        values['CIFAR-100', method] = [100 * cifar[str(s)][key + '_leaf_acc'] for s in (1, 2, 3)]
        first_key = {'flat': 'flat_fedavg_leaf_acc', 'tree': 'fedavg_tree_leaf_acc', 'resolvfl': 'resolvfl_leaf_acc'}[key]
        values['iNat-Aves', method] = [100 * first[first_key]] + [100 * inat[str(s)][key]['leaf_acc'] for s in (2, 3)]
        im_key = {'flat': 'flat', 'tree': 'unmasked', 'resolvfl': 'masked'}[key]
        values['ImageNet-3L/F', method] = [imagenet[str(s)][im_key]['leaf_acc'] for s in (1, 2, 3)]
    for (dataset, method), observations in values.items():
        matches = [r for r in rows if r['dataset'] == dataset and r['method_or_endpoint'] == method and r['experiment'] == 'main_comparison' and r['metric'] == 'leaf_accuracy']
        assert len(matches) == 1, (dataset, method)
        r = matches[0]
        mean, sd = statistics.mean(observations), statistics.stdev(observations)
        assert abs(mean - float(r['mean'])) <= 0.00501, (dataset, method, mean)
        assert abs(sd - float(r['sd'])) <= 0.00501, (dataset, method, sd)
        print(f'{dataset}: {method}: {mean:.2f} +/- {sd:.2f}: matches')
    print('PASS: source hashes and all 9 main mean/sample-SD pairs. No training executed.')

if __name__ == '__main__':
    main()

