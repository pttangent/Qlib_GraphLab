# NFF-only stock-day episode and generic-model pilot

This pilot adds two independently runnable stages on top of the existing
`qlib.contrib.data.nff` adapter. It does not read GFF or GAL.

## P1 — StockDay Episode Factory

Each `(symbol, trade_date)` becomes one episode:

```text
support: 09:30 <= conservative decision_time < 10:00
queries: 10:00, 10:15, ..., 15:00
input:   the last 30 decision-time minutes ending at each query
labels:  future open-to-open returns and close-based realized volatility
```

The existing NFF adapter remains authoritative for the availability clock:

```text
decision_time = ceil(max(upstream available_time), 1min) + delay_bars * 1min
```

P1 reads NFF Parquet directly and writes only manifests, rejection audits and
metadata. It does not create another wide warehouse copy.

### Commands

```powershell
python examples/nff_stockday_episode.py inspect `
  --config configs/nff_stockday_generic_pilot.yaml

python examples/nff_stockday_episode.py build-manifest `
  --config configs/nff_stockday_generic_pilot.yaml

python examples/nff_stockday_episode.py inspect-episode `
  --config configs/nff_stockday_generic_pilot.yaml `
  --symbol NVDA `
  --date 2026-05-08 `
  --output D:\TEMP\NVDA_2026-05-08_episode.npz
```

### Checkpoints

P1 checkpoints by complete trading date:

```text
<episode_run_root>/
├── run_contract.json
├── status.json
├── dates/date=YYYY-MM-DD/
│   ├── manifest.parquet
│   ├── rejections.parquet
│   └── meta.json
└── aggregate/
    ├── episode_manifest.parquet
    ├── episode_rejections.parquet
    ├── episode_daily_summary.parquet
    ├── summary.json
    └── report.md
```

A date is reused only when the contract hash matches the source schemas,
execution policy, support/query policy, features and targets.

## P2 — Generic shared GRU baseline

P2 is the B0 baseline and deliberately has no adaptation:

```text
30 x F causal query window
  -> train-only standardization
  -> append element observation mask (30 x 2F)
  -> shared GRU
  -> multi-target regression head
```

There is no ticker embedding, daily adapter, GFF context, GAL label source or
online gradient update. P2 therefore tests whether a common cross-stock
representation exists in NFF alone.

### Commands

P1 must finish first because P2 uses its aggregate manifest to establish the
available trading dates.

```powershell
python examples/nff_generic_baseline.py train `
  --config configs/nff_stockday_generic_pilot.yaml

python examples/nff_generic_baseline.py evaluate `
  --config configs/nff_stockday_generic_pilot.yaml
```

The default split uses complete dates only: first 70% train, next 15% validation,
final 15% test. Normalization is fitted from training dates only.

### Checkpoints and outputs

```text
<model_run_root>/
├── train_contract.json
├── normalization.json
├── normalization_progress.json
├── checkpoint_last.pt
├── checkpoint_best.pt
├── training_history.parquet
├── summary.json
├── report.md
└── test/
    ├── predictions/date=YYYY-MM-DD.parquet
    ├── predictions_all.parquet
    ├── daily_metrics.parquet
    └── summary.json
```

Training checkpoints after every trading date and records the optimizer, AMP
scaler, epoch, next date index and contract hash.

## Acceptance gates

P1 must prove:

- support ends strictly before the first query;
- every input sequence ends at or before its query timestamp;
- targets remain within the regular session;
- winter and daylight-saving dates use `America/New_York` correctly;
- incomplete or mixed NFF contracts are rejected;
- every rejected episode has an explicit reason.

P2 must prove:

- temporal train/validation/test segments are strictly ordered;
- feature and target normalization use training dates only;
- training resumes at the next unfinished date;
- predictions are emitted for every observed target;
- reports contain MSE, MAE and cross-sectional RankIC metrics;
- no GFF or GAL path is accessed.

## Automated validation

The branch workflow runs the P1/P2 tests, the existing NFF adapter regression
tests and Python compilation on both Ubuntu and Windows. This verifies the
causal episode contract, model tensor contract and Windows-compatible entrypoints.

A later P3 may add a support encoder and daily adapter, but it must compare
against P2 using exactly the same query samples and temporal test dates.