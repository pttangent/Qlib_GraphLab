# NFF research v2.4 CPU fast path

## Purpose

The v2.3 runner was healthy but CPU-bound: all workers were active, memory and
storage were stable, and the dominant cost was repeated Pandas rank/groupby and
small Ridge setup.  The v2.4 fast path keeps the v2.3 formulas and output
contracts while replacing repeated execution work.

Reference commit: `5178346e93bcb3a4ebd90adaec8f98211112aaaa`.

Entry point:

```powershell
python nff_research/v2_4_optimized_runner.py `
  --config configs/v2_4_optimized_full.yaml `
  --run-id smoke_v2_4_20260803 `
  --start-date 2026-01-02 `
  --end-date 2026-01-06
```

Use a new output/run ID.  Do not resume a v2.3 output directory because the run
contract and source projection differ.

## Optimization contract

### 1. Exact source projection

`research_required_columns_manifest()` derives warehouse columns from:

- the 55 representative alpha fields;
- five Hawkes gate fields;
- active/stale eligibility controls;
- bar and trade columns needed for labels and neutralization.

It attempts to read a materialized `hawkes_derived` family first.  When the
family or required columns are unavailable, only the narrow Hawkes Lite
fallback dependencies are loaded and the existing derived formulas are used.
Deprecated `signal__` columns are not materialized.

The resolved schemas, exact column list, and count are written into each daily
`optimization_profile`.

### 2. One IC rank pass

The reference runner separately ranked the same feature block for minute-mean
and pooled IC.  The fast path ranks feature and label columns once per decision
minute, then:

- records the minute Spearman correlation;
- updates online pooled covariance state for corrected demeaned percentile-rank
  IC.

No day-sized ranked DataFrame is retained for pooled statistics.

### 3. Residual feature reuse

Open/open, VWAP/VWAP, and close/close return labels frequently share an
identical eligible index for the same universe and horizon.  Residualized
feature matrices are cached by universe plus exact index identity.  Label
residuals remain separate.  Samples with different indices never share a
residual matrix.

### 4. Batched OOF setup

For each admitted 15-minute cross-section the fast path computes once:

- the all-step common complete-case sample;
- the ranked and demeaned full feature matrix;
- stable SHA256 symbol fold IDs;
- each fold's training-label rank.

The four cumulative bundle steps then use column views of the same matrix.
Output metrics and cross-sectional OOF semantics are unchanged.

### 5. Early portfolio and decile pruning

Portfolio and decile inputs are reduced to regular-session 15-minute decision
points before the reference groupby loops.  A complete 390-minute session is
reduced to 26 timestamps.  Existing 30/60-minute variants remain subsets of
those timestamps and retain their reference sleeve accounting.

### 6. LPT date scheduling

Daily warehouse parquet bytes are cached in:

```text
NFF_research/derived_inputs/research_date_cost_v2_4.json
```

Dates are launched longest-estimated-processing-time first.  This follows the
NFF family-build scheduling pattern and reduces the final campaign tail.

### 7. Thread discipline and profiling

Date-level processes remain the parallelism boundary.  Unless the user already
set a value, workers set the following to one thread:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
```

Each daily `meta.json` gains:

```json
{
  "optimization_profile": {
    "fastpath_version": "2.4",
    "total_seconds": 0,
    "stage_seconds": {
      "feature_build": 0,
      "labels": 0,
      "controls": 0,
      "ic_and_neutralization": 0,
      "deciles": 0,
      "oof": 0,
      "portfolio": 0
    },
    "required_columns": {},
    "symbol_fold_cache": "...",
    "scheduler": "LPT file-byte estimate"
  }
}
```

## Equality and safety tests

`tests/test_nff_research_v2_4_fastpath.py` checks:

- minute-mean and pooled IC equality against the reference implementation;
- batched OOF prediction equality for multiple cumulative steps;
- online pooled correlation equality with missing observations;
- 390 to 26 minute prefiltering before portfolio groupby;
- deterministic LPT ordering;
- narrow source projection and materialized Hawkes derived reuse;
- BLAS thread guards without overriding explicit user settings.

Both Windows and Linux CI compile and run the fast-path tests.

## Smoke acceptance

Run at least one winter date and one daylight-saving date.  Compare the same
dates against v2.3 using separate output roots.

Required correctness checks:

- identical factor IC rows within floating tolerance;
- identical OOF prediction metrics within floating tolerance;
- identical portfolio rows after sorting by the full key;
- identical label counts and universe counts;
- no mixed run-contract reuse.

Performance fields to compare:

- total seconds per date;
- IC and neutralization seconds;
- OOF seconds;
- portfolio seconds;
- peak worker RSS;
- projected source column count.

Targets, not guarantees:

- source columns materially below the previous 147-field load;
- portfolio groupby input minutes reduced from 390 to 26;
- IC rank work approximately halved;
- OOF feature-ranking and fold hashing reduced from once per step to once per
  minute;
- median daily wall time reduced by at least 30%;
- no increase in numerical differences beyond test tolerance.

If numerical equality fails, use the v2.3 reference runner and treat the
fast path as rejected.  Do not relax formulas to obtain speed.
