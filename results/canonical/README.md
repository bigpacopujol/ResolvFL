# Manuscript result registry

`manuscript_results.csv` contains the numerical results used by the current Pattern Recognition manuscript.

Each row records:

- dataset and experiment;
- method or evaluated endpoint;
- metric family and evaluation scope;
- mean/sample SD or seed-level value;
- provenance level;
- manuscript eligibility.

The JSON export contains the same manuscript-facing registry in machine-readable form. `registry_status.json` records the registry version and generation date.


The consolidated MASTER dated 2026-10-08 defines acceptance. There are 76 accepted records; four unpromoted iNat summaries are archived in `../quarantine/`. Their exclusion does not alter the accepted main results or intervention endpoints. `run_index` is presentation order, not a random seed. Missing run identifiers are not reconstructed from rounded values.
