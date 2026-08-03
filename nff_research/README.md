# NFF research runners

This directory freezes the local research source used for the NFF v2.1
neutralized full-run validity pass.

The current runner is:

- `v2_1_neutralized_runner.py`

The matching run configuration is:

- `configs/v2_1_neutralized_full.yaml`

The runner intentionally lives outside the core `qlib.contrib.data.nff` adapter
surface. It is a reproducible research engine that consumes the local NFF
warehouse, builds PIT daily controls, evaluates representative NFF features
with both minute-mean and pooled RankIC, performs return-label neutralization,
and emits a 15-minute staggered long-short portfolio proxy.

Known limits:

- The paths in the v2.1 config are local workstation paths.
- Sector neutralization is not applied because no local sector reference file
  was available at run time.
- The portfolio output is a transparent proxy, not a full Qlib Recorder
  strategy backtest.
