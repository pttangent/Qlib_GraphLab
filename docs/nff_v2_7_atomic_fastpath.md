# NFF v2.7 real-schema atomic fast path

## Purpose

This branch optimizes the full-defined NFF minute-factor campaign without
changing the research questions. The rebuilt `nvg_supplement` is a first-class
input. Every date audits the actual Parquet schema before derivation; missing
fields are recorded and are never silently replaced by similarly named fields.

Authoritative upstream source:

- Repository: `pttangent/NodeFactorFactory`
- Branch: `agent/nvg-trade-hawkes-families`
- Warehouse supplement namespace: `nvg_supplement/*/schema=v1/date=YYYY-MM-DD`

## Directional NVG source

The preferred direction fields come from
`nvg_supplement/minute_nvg_edge_raw`, including, per available minute window:

- `price_nvg_{W}m_terminal_signed_edge_balance`
- `price_nvg_{W}m_terminal_long_edge_signed_slope`
- `price_detrended_nvg_{W}m_terminal_signed_edge_balance`
- `price_detrended_nvg_{W}m_terminal_long_edge_signed_slope`
- `volume_nvg_{W}m_terminal_signed_edge_balance`
- `volume_nvg_{W}m_terminal_long_edge_signed_slope`
- `price_volume_nvg_{W}m_edge_weighted_jaccard`
- `price_volume_nvg_{W}m_common_edge_slope_corr`

The runtime schema, not this documentation, determines which windows and
columns are executable for a date. Base `minute_nvg` top/bottom geometry remains
available for geometric factors, but it is not substituted for a missing rebuilt
directional field.

## Runtime resolution audit

Every date writes:

```text
atomic_checkpoints/date=YYYY-MM-DD/schema/
  nff_schema_audit.parquet
  nff_schema_audit.csv
  field_resolution.parquet
  field_resolution.csv
  factor_resolution.parquet
  factor_resolution.csv
  factor_resolution_summary.json
```

`field_resolution` records each requested name, the exact base/supplement field
or documented alias used, call count and non-null rate. `factor_resolution`
records every formal factor specification, formula, window, materialized column,
non-null rate and failure state. Therefore a campaign cannot report “476
factors” without showing which specifications actually computed.

## Causal path reconstruction

Some financial prototypes require path metrics that are mathematically defined
from canonical minute bars but are not necessarily persisted by the older NFF
family. v2.7 reconstructs them causally per symbol and window:

```text
signed_change     = x_t - x_(t-W+1)
path_length       = sum(|x_i - x_(i-1)|)
efficiency        = |signed_change| / (path_length + eps)
roughness         = path_length / (|signed_change| + eps)
range             = rolling_max(x) - rolling_min(x)
terminal_position = (x_t - rolling_min(x)) / (range + eps)
```

Price paths use log close; volume paths use `log1p(dollar_volume)`. No future
minute is used.

## Mathematical corrections

The v2.7 patch fixes the following definitions:

1. `C08`: new-hub direction uses the sign of the **change** in asymmetry,
   multiplied by hub-replacement strength.
2. `F03`: stale-to-active transition uses per-symbol positive decreases in
   stale ratio times per-symbol positive increases in activity.
3. `G09`: irreversibility exhaustion uses the decline from the five-minute
   lagged value and points against the existing momentum direction.
4. `H23`: effective duration uses the effective-duration field rather than
   persistence.
5. `I06`: large-trade direction equals large-buy/sell imbalance multiplied by
   large-trade dollar share.
6. Venue HHI and entropy use actual venue volume shares rather than `1/N` and
   `log(N)` proxies.
7. Decile boundaries are vectorized but remain equivalent to `qcut` applied to
   unique within-minute ranks.
8. Account replay interprets model output as expected future return: high
   predictions are long and low predictions are short.

## Exact-minute labels

The v2.6 exact-time formula remains authoritative. v2.7 caches each
`(daily frame, source column, future offset)` lookup, so all horizons reuse the
same exact-time result instead of repeating the same reindex/join.

Entry is the exact next minute (`t+1`) and exit is `t+h+1`. Missing intermediate
minutes invalidate future-window labels rather than stretching the holding
period over the next available row.

## Atomic checkpoints

Checkpoints are stored under:

```text
<run>/atomic_checkpoints/date=YYYY-MM-DD/
```

The hierarchy is:

```text
schema/
factors/family=<A-K>/window=<window>/block=<NNN>.parquet
neutralization/residual=<contract>.parquet
stages/decile_curves.parquet
stages/portfolio_proxy.parquet
stage_events.jsonl
```

A factor-window manifest includes the run contract, instruction hash, index
hash, factor IDs, formulas and block files. An interrupted run reuses only
blocks whose complete contract matches.

## Parallel architecture

The default half-year configuration uses:

```text
2 date processes × 8 intra-date shared-memory workers ≈ 16 CPU slots
```

A date process retains a wide symbol-minute frame. Starting 16 date processes
would duplicate that frame and can multiply an 8–27 GB peak into an unsafe
memory demand. Intra-date threads share the daily frame and parallelize
decile/factor work without that duplication.

The outer scheduler may increase to three dates only when its RSS and available
memory guards allow it.

## Main performance changes

- factor families are derived once per `(family, window)` and written in small
  reusable blocks;
- exact future-minute values are cached by source/offset;
- residualized factor matrices are checkpointed by contract;
- deciles use grouped unique ranks and NumPy `bincount` aggregation instead of a
  Python loop over every minute-factor-decile cell;
- only Alpha outputs plus the small universe/control set remain in the main
  frame after derivation;
- train-only redundancy filtering uses IC preselection, a deterministic bounded
  sample and one correlation matrix;
- stage timing and state transitions are written continuously.

## Walk-forward models

The chronological train/validation/test loop supports:

- Equal Weight;
- training-IC Weight;
- Ridge;
- ElasticNet;
- LightGBM when installed;
- a real PyTorch GRU when installed.

Feature coverage, direction, redundancy filtering, model parameters and the
primary model are chosen using train/validation only. Missing optional packages
produce `MODEL_UNAVAILABLE`; they are not silently replaced. Ridge family
ablation removes each A–K/S family before feature selection and retrains the
model on the same chronological fold.

## Running

```powershell
python nff_research/v2_7_launch.py `
  --config configs/v2_7_atomic_full_campaign.yaml
```

Use a new run ID when changing factor formulas, source schemas, labels,
neutralization, costs, universes or walk-forward settings. Worker/dashboard
settings do not change the semantic contract unless they alter numerical
results.

## Validation requirement

Before the half-year run, execute one full date and inspect:

- `nff_schema_audit.csv` has zero required directional-field omissions for the
  intended windows;
- `factor_resolution_summary.json` shows the expected successful factor count
  and explains every unavailable specification;
- factor-block manifests are reusable after a forced process interruption;
- v2.7 and v2.6 exact labels match on complete-minute samples;
- vectorized and reference deciles match within floating-point tolerance;
- peak RSS remains compatible with the selected outer date parallelism.

A smoke run is not a completed research campaign. Temporal OOS, Recorder,
orders, fills, positions, account equity, family ablation and capacity outputs
must still pass the campaign completion contract.
