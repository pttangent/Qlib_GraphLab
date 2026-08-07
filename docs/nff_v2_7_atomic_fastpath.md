# NFF v2.7 warehouse-exact atomic fast path

## Purpose

This branch optimizes the full-defined NFF minute-factor campaign while making
the research contract match the physical warehouse. The rebuilt
`nvg_supplement` is a first-class input. Every date audits the actual Parquet
schema, field resolution, physical join, factor coverage and checkpoint
contract before a date can complete.

Authoritative upstream source:

- Repository: `pttangent/NodeFactorFactory`
- Branch: `agent/nvg-trade-hawkes-families`
- Warehouse root: `D:/DEV/AnotherNetworkFactory/warehouses/NFF_warehouse`
- Supplement namespace: `nvg_supplement/*/schema=v1/date=YYYY-MM-DD`

## Physical windows and formal contract

The 2026-08-06 warehouse inventory defines these physical products:

| Layer | Physical windows |
|---|---|
| base minute NVG/path | `10m / 15m / 30m` |
| minute NVG edge supplement | `10m / 15m / 30m` |
| minute NVG topology supplement | `10m / 15m / 30m` |
| minute HVG risk/topology | `15m / 30m / 60m` |
| trade visibility | `60s / 180s / 300s` |
| formal S directional factors | `15m / 30m` |

The historical instruction expands to **474 A–K specifications plus two S
specifications = 476 formal specifications**. The physical warehouse supports
**462 A–K specifications plus two S factors = 464 executable specifications**.
The difference is exactly the twelve `C@60m` price-NVG-topology
specifications. No 60m price-NVG topology product exists, so those twelve rows
are retained in `legacy_contract_gap` and are never filled by renaming 30m data
or substituting 60m HVG risk fields.

The production model, IC, decile and ablation stages consume only the 464
warehouse-exact specifications. The 476-row registry remains available for
legacy instruction reconciliation.

## Directional NVG source

Terminal direction comes from `nvg_supplement/minute_nvg_edge_raw` and
`trade_visibility_edge_raw`:

- `*_terminal_signed_edge_balance`;
- `*_terminal_long_edge_signed_slope`;
- raw and detrended price direction;
- volume direction;
- price/volume edge-weighted Jaccard and common-edge slope correlation.

Base top/bottom geometry remains available for geometric factors, but
`top_bottom_asymmetry` and unsigned `long_edge_ratio` are not substitutes for a
missing terminal direction field.

The formal S family contains only:

```text
full_factor__s01__w15m
full_factor__s01__w30m
```

The raw 10m direction fields remain available for A–K dependencies and audit;
they do not create an unregistered S10 model input.

## Physical join and PIT clock

The primary adapter aligns canonical and feature sources on
`(symbol_id, timestamp)`. v2.7 preserves the exact `symbol_id` and original
`event_time` as temporary research metadata, then joins supplement rows on
`(symbol_id, event_time)`.

A supplement value is admitted only when:

```text
ceil(supplement.available_time, 1 minute) + 1 minute <= decision_time
```

A matched row that becomes available later is set to missing at the current
decision row; it is not silently shifted or backfilled. Production runs fail
if the exact metadata is unavailable. Each date writes
`supplement_join_audit.{parquet,csv,json}` with matched, admissible, late,
duplicate and fallback counts.

## Runtime resolution audit

Every date writes under:

```text
atomic_checkpoints/date=YYYY-MM-DD/schema/
```

including:

```text
nff_schema_audit.parquet
field_resolution.parquet
factor_resolution.parquet
factor_resolution_summary.json
physical_factor_completion_gate.json
legacy_contract_gap.parquet
legacy_contract_gap.json
supplement_join_audit.parquet
supplement_join_audit.json
```

`field_resolution` records each requested field and the exact physical column
or documented mathematical reconstruction used. `factor_resolution` records
formula, window, source fields, materialized column, non-null rate and failure
state. The physical completion gate requires all 464 executable specifications
to be present and non-zero-coverage; the twelve legacy-only C@60m rows are
reported separately and do not count as an execution failure.

## Causal reconstruction

Path fields already published in `features/minute_nvg` remain the preferred
source. When a mathematically identical minute-bar path dependency is absent,
v2.7 can reconstruct it causally per symbol and window:

```text
signed_change     = x_t - x_(t-W+1)
path_length       = sum(|x_i - x_(i-1)|)
efficiency        = |signed_change| / (path_length + eps)
roughness         = path_length / (|signed_change| + eps)
range             = rolling_max(x) - rolling_min(x)
terminal_position = (x_t - rolling_min(x)) / (range + eps)
```

Price paths use log close; volume paths use `log1p(dollar_volume)`. Event-time
trade paths are not reconstructed from one-minute aggregates when the original
event sequence is required.

## Mathematical corrections

The final formula chain includes:

1. `A14–A16`: multi-scale momentum uses physical `10m/15m/30m`.
2. `B04`: long-horizon direction uses terminal long-edge signed slope, not an
   unsigned long-edge ratio.
3. `C08`: new-hub direction uses the sign of the change in asymmetry multiplied
   by hub-replacement strength.
4. `D04`: abnormal volume expansion is a causal within-symbol time-series
   z-score.
5. `E01–E22`: direction-sensitive trade formulas use exact terminal signed
   direction fields; E20/E21 do not fall back to top/bottom asymmetry.
6. `F03`: stale-to-active transition uses per-symbol positive stale decreases
   and activity increases.
7. `G09`: irreversibility exhaustion points against existing momentum when
   irreversibility declines from its five-minute lag.
8. `H23`: effective duration uses the duration field, not persistence.
9. `I06`: large-trade direction is large-buy/sell imbalance times large-trade
   dollar share.
10. K minute-price/topology anchors remain the documented physical 10m anchor;
    trade/Hawkes anchors retain their specified horizons.
11. Venue HHI and entropy use actual venue volume shares.
12. Account replay is high-prediction long and low-prediction short.

## Exact-minute labels

Entry is the exact next minute (`t+1`) and exit is `t+h+1`. Missing required
minutes invalidate the label rather than stretching the holding period over the
next observed row. Exact future lookups are cached by source column and offset;
the cache changes performance, not label semantics.

## Atomic checkpoints

The hierarchy includes:

```text
factors/family=<A-K>/window=<window>/block=<NNN>.parquet
deciles/universe=<...>/label=<...>/variant=<...>/block=<NNN>.parquet
neutralization/residual=<contract>.parquet
stages/decile_curves.parquet
stage_events.jsonl
```

Factor and decile blocks include the run contract, instruction hash, index
hash, formula/version information and an upstream Parquet fingerprint. If a
base or supplement Parquet file changes, stale blocks are not reused.

## Parallel architecture

The default half-year configuration uses:

```text
2 date processes × 8 intra-date shared-memory workers ≈ 16 CPU slots
```

Outer date processes own wide daily frames; intra-date threads share a daily
frame. The scheduler may increase to three dates only when RSS and available
memory guards permit it.

## Walk-forward models

The chronological train/validation/test loop supports:

- Equal Weight;
- training-IC Weight;
- Ridge;
- ElasticNet;
- LightGBM when installed;
- a real PyTorch GRU when installed.

Feature coverage, direction, redundancy filtering, model parameters and primary
model selection use train/validation only. Optional unavailable packages are
recorded as `MODEL_UNAVAILABLE`. A–K/S family ablation removes a family before
feature selection and retrains on the same chronological fold.

## Running

```powershell
python nff_research/v2_7_launch.py `
  --config configs/v2_7_atomic_full_campaign.yaml
```

The current configuration uses run name:

```text
v2_7_atomic_full_defined_20260806_r2
```

Use a new run ID after changing formulas, physical schemas, labels,
neutralization, universes, costs or walk-forward settings.

## Validation requirement

Before the half-year campaign:

1. run one complete date against the local warehouse;
2. verify `physical_factor_completion_gate.json` reports 464 successful
   executable specifications;
3. verify `legacy_contract_gap.json` contains exactly twelve C@60m rows;
4. verify the supplement join has no duplicate keys or symbol fallback;
5. interrupt and resume after at least one factor and decile block;
6. compare exact labels and vectorized deciles with their reference paths;
7. inspect peak RSS and elapsed time.

A CI pass or smoke run is not a completed half-year research campaign. The PR
remains draft until the real one-date benchmark and interrupted-resume test are
run on the local warehouse.
