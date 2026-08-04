# NFF-only context decomposition pilot

This P4 branch extends the completed P1-P3 pipeline without reading GFF or GAL.
It decomposes prediction context into four independently testable sources:

```text
B0  query-local NFF state                         (existing P2)
B1  B0 + same-stock same-day opening support     (existing P3)
B2  B0 + prior-20-day stock identity             (P4 identity)
B3  B0 + current leave-one-out market state      (P4 market)
B4  B0 + identity + market                       (P4 combined)
B5  B0 + identity + market + daily support       (P4 full)
```

All variants use the same P1 sample contract, P2 normalization, P2 checkpoint,
complete-date temporal split and `(sample_id, target_name)` keys.

## Stage A — context cache

The cache is deliberately small. It is not a second NFF warehouse.

### Stock identity rows

One row is stored per `(symbol, trade_date)`. For every NFF feature it records:

- daily mean;
- daily standard deviation;
- last observable value;
- feature coverage.

The daily summary contains only NFF features. A row dated `d` may only be used
for episodes dated after `d`; the model loader rejects current/future history.
The default identity sequence uses the previous 20 available trading days.

### Market rows

One row is stored per P1 query sample. At each query timestamp the cache uses
the latest causally available NFF feature snapshot for all admitted stocks and
computes exact leave-one-out:

- mean;
- standard deviation;
- positive ratio;
- coverage.

The target stock is removed from its own market context. No end-of-day market
statistics or future query rows are used.

### Commands

```powershell
python examples/nff_context_cache.py inspect `
  --config configs/nff_context_decomposition_pilot.yaml

python examples/nff_context_cache.py build `
  --config configs/nff_context_decomposition_pilot.yaml
```

Checkpoints are date partitions:

```text
<NFF_context_cache>/runs/context_v1/
├── context_contract.json
├── dates/date=YYYY-MM-DD/
│   ├── identity.parquet
│   ├── market.parquet
│   └── meta.json
└── aggregate/
    ├── context_daily_audit.parquet
    ├── summary.json
    └── report.md
```

A completed date is reused only when the episode, normalization and context
formula hashes match.

## Stage B — context models

The P2 GRU and head are frozen in the primary P4 experiment. Each context has a
separate encoder, zero-initialized residual and gate:

```text
h = h_query
  + identity_gate * identity_delta
  + market_gate   * market_delta
  + daily_gate    * daily_delta
```

Zero initialization makes every new variant equal P2 before training. Context
dropout is applied only to active context branches, so the model can fall back
to P2 when a context is unavailable.

### Train B2-B5

```powershell
python examples/nff_context_model.py train --variant identity `
  --config configs/nff_context_decomposition_pilot.yaml

python examples/nff_context_model.py train --variant market `
  --config configs/nff_context_decomposition_pilot.yaml

python examples/nff_context_model.py train --variant combined `
  --config configs/nff_context_decomposition_pilot.yaml

python examples/nff_context_model.py train --variant full `
  --config configs/nff_context_decomposition_pilot.yaml
```

Each variant checkpoints after every completed training date. Evaluation can be
re-run without retraining:

```powershell
python examples/nff_context_model.py evaluate --variant full `
  --config configs/nff_context_decomposition_pilot.yaml

python examples/nff_context_model.py evaluate-all `
  --config configs/nff_context_decomposition_pilot.yaml
```

## Counterfactual controls

Identity variants evaluate same, shuffled-stock and zero identity. Market
variants evaluate same-time, shuffled-query-time, previous-day and zero market
context. The full model additionally evaluates shuffled-stock, previous-day and
zero daily support.

Every report is paired against the exact P2 prediction for the same sample.
Bootstrap resampling uses complete trading dates, not individual minute rows.

## Acceptance gates

P4 is accepted only when:

- identity history dates are strictly earlier than the episode date;
- market context is leave-one-out and uses the current query clock only;
- P2 parameters remain frozen;
- zero-initialized variants reproduce P2 exactly;
- all B2-B5 predictions preserve unique paired keys;
- context caches resume by date under an exact contract hash;
- Windows and Linux tests pass;
- no GFF or GAL path is accessed.

The key research comparison is `B5 - B4`. A positive and stable B5 increment
means same-stock same-day support contains information not already explained by
persistent stock identity or the current NFF market state.
