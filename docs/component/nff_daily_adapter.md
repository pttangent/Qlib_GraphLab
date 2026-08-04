# NFF-only P3 daily context adapter

P3 extends the completed P1/P2 pilot without reading GFF or GAL. Its only
research question is whether a fixed early-session NFF support block improves
later predictions on the exact same P2 query samples.

## Model contract

```text
09:30-10:00 NFF support -> support GRU -> daily context
30-minute causal query  -> frozen P2 GRU -> query representation
(context, query)        -> zero-initialized gated residual -> P2 head
```

The residual output layer starts at zero. Before training, P3 therefore equals
P2 exactly. During the primary experiment the full P2 backbone and prediction
head remain frozen and in evaluation mode, including dropout. Only the support
encoder, residual adapter and gate train.

P3 uses no ticker embedding, GFF context, GAL label source, or intraday gradient
update.

## Stable paired samples

Every query receives a deterministic `sample_id` built from:

- symbol and trade date;
- query timestamp;
- P1 episode contract hash;
- ordered feature names;
- ordered target names.

The primary evaluation refuses to finish unless `(sample_id, target_name)` is
unique and both P2 and P3 predictions exist for every row. Reports also retain
`query_index` and `support_query_overlap_minutes`, so the 10:00 overlap-heavy
queries cannot hide degradation later in the session.

## Commands

P1 and P2 must already be complete.

```powershell
# Re-export the P2 checkpoint with the P3 paired sample contract.
python examples/nff_daily_adapter.py export-paired-baseline `
  --config configs/nff_stockday_adapter_pilot.yaml

# Primary adapter-only training. Checkpoints after every trading date.
python examples/nff_daily_adapter.py train `
  --config configs/nff_stockday_adapter_pilot.yaml

# Paired temporal-OOS evaluation and all counterfactuals.
python examples/nff_daily_adapter.py evaluate `
  --config configs/nff_stockday_adapter_pilot.yaml
```

Optional upper-bound experiment:

```powershell
python examples/nff_daily_adapter.py train-joint `
  --config configs/nff_stockday_adapter_pilot.yaml

python examples/nff_daily_adapter.py evaluate-joint `
  --config configs/nff_stockday_adapter_pilot.yaml
```

Joint training unfreezes only the P2 head, LayerNorm and final GRU layer. It is
not the primary personalization evidence.

## Counterfactual support modes

Evaluation uses the same trained adapter and identical query samples:

| Mode | Meaning |
|---|---|
| `same_stock_same_day` | True stock-day support |
| `same_day_shuffled_stock` | Another stock from the same date |
| `same_stock_previous_day` | Same stock's most recent earlier test-date support |
| `zero_support` | Architecture/no-information control |

Interpretation:

- same-stock > shuffled-stock: stock-day-specific information;
- same-stock ~= shuffled-stock > zero: common daily market state;
- previous-day ~= same-day: persistent stock identity dominates;
- same-stock <= zero: stop the daily-adaptation path.

The support-length ablation evaluates 5, 15 and 30 opening minutes without
rebuilding P1 episodes.

## Checkpoints and outputs

```text
<adapter_run_root>/
├── adapter_contract.json
├── joint_contract.json                         # optional
├── paired_sample_contract.json
├── paired_baseline_predictions.parquet
├── checkpoint_adapter_last.pt
├── checkpoint_adapter_best.pt
├── checkpoint_joint_last.pt                    # optional
├── checkpoint_joint_best.pt                    # optional
├── training_history_adapter.parquet
├── training_history_joint.parquet              # optional
├── summary.json
├── report.md
└── test/
    ├── predictions/date=YYYY-MM-DD.parquet
    ├── predictions_b0_vs_adapter.parquet
    ├── daily_metrics.parquet
    ├── target_metrics.parquet
    ├── query_time_metrics.parquet
    ├── support_overlap_metrics.parquet
    ├── support_ablation_metrics.parquet
    ├── adaptation_gate_distribution.parquet
    ├── sample_parity_audit.json
    ├── paired_date_bootstrap.json
    └── summary.json
```

The bootstrap samples complete trading dates, not individual minute rows.

## Acceptance gates

Engineering gates:

1. P2/P3 paired keys are 100% identical and unique.
2. Support ends strictly before every query.
3. Normalization and temporal splits are reused from P2.
4. Shuffling never crosses a trading date or temporal segment.
5. Adapter training checkpoints after every completed date.
6. Frozen P2 dropout remains disabled during adapter-only training.
7. Windows and Linux tests pass.

Research gates:

1. Improvement remains after 10:00 overlap-heavy queries are separated.
2. Gains occur across dates rather than a few isolated sessions.
3. Same-stock support beats shuffled and zero-support controls.
4. Return gains do not require a large volatility-target degradation.
5. Gate behavior is reported rather than assumed useful.
