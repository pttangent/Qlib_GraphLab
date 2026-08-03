# NFF research runners

This directory freezes the local research source used for the NFF v2.2
neutralized OOF incremental validity pass.

The current runner is:

- `v2_1_neutralized_runner.py`

The matching run configuration is:

- `configs/v2_1_neutralized_full.yaml`

The runner intentionally lives outside the core `qlib.contrib.data.nff` adapter
surface. It is a reproducible research engine that consumes the local NFF
warehouse, builds PIT daily controls on the New York regular session
(`09:30 <= local_time < 16:00`), evaluates representative NFF features with
both minute-mean and corrected pooled RankIC, performs return-label
neutralization, validates feature bundles with fixed-symbol-fold
cross-sectional OOF Ridge on one common sample, and emits a same-sleeve
long-short portfolio proxy. VWAP-to-VWAP is the primary execution result;
open-to-open remains diagnostic.

Known limits:

- The runner filename is retained for compatibility; contracts and reports
  explicitly identify research version 2.2 and base runner version 2.1.
- The paths in the config are local workstation paths.
- Sector neutralization is not applied because no local sector reference file
  was available at run time.
- The portfolio output is a transparent sleeve-accounting proxy, not a full
  Qlib Recorder strategy backtest.
- Fixed-symbol-fold OOF tests same-time cross-sectional generalization, not
  future-date generalization. A temporal walk-forward stage is still required.
- Each run writes `run_contract.json`; existing daily checkpoints are reused
  only when the date-level `meta.json` carries the same contract hash.
