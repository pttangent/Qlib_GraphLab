# NFF v2.8 Pipelined Selection

## Why this exists

The v2.7 resource snapshot showed that one date process retained the same wide
frame through stages with very different cost profiles:

- factor derivation: about 156 seconds in the completed benchmark date;
- IC and neutralization: about 2,354 seconds;
- portfolio proxy: about 3,593 seconds;
- three simultaneous IC workers reached roughly 94 GB RSS.

A date-only scheduler can cap concurrency, but it cannot release the wide frame
between stages or allow a fast stage to run ahead of a slow stage. v2.8 changes
the execution unit from `date` to `stage x date`.

## DAG

```text
materialize
    |
    v
basic_screen (all 464 physical factors, raw IC only)
    |
    v
select (first 60 completed training dates only)
    |                         |
    v                         v
detailed                  portfolio
<=120 candidates          <=40 candidates
neutralization            frozen direction
multi-label IC            Hawkes gate pairs
deciles                   costs/turnover
```

## Stage contracts

### 1. materialize

Loads the NFF warehouse, merges rebuilt supplements, derives all 464 physical
factors, builds exact-minute labels and controls, and writes:

- v2.7 family/window/factor-block checkpoints;
- support columns;
- labels and PIT masks;
- controls;
- already-resolved universe masks;
- the feature registry and factor-block inventory.

The full wide frame is then released. Factor values are not duplicated into a
second monolithic file; downstream stages read only required v2.7 factor blocks.

### 2. basic_screen

Streams all factor blocks and computes cheap raw rank-IC diagnostics for every
physical factor. It does not build residual matrices, deciles or portfolios.
The configured default preserves all return, state, jump and mechanical-cost
labels in the raw screen.

### 3. select

Uses only the first 60 completed screen dates. For each factor it chooses the
best configured horizon and records:

- mean daily rank IC;
- IC standard deviation and ICIR;
- valid days;
- mean coverage;
- daily and minute-level direction consistency;
- a deterministic selection score;
- the frozen direction sign.

Family minimums and maximums prevent a single highly redundant family from
occupying the entire candidate list.

Defaults:

- 120 detailed-diagnostic candidates;
- 40 portfolio candidates;
- at least two candidates per family where available.

### 4. detailed

Loads only the detailed candidates. Raw and neutralized multi-label IC and
candidate decile curves are calculated with the v2.7 formulas and exact-minute
labels. v2.7 residual and decile checkpoint contexts remain active.

### 5. portfolio

Loads only portfolio candidates and multiplies every candidate by its frozen
training direction. The portfolio engine therefore always interprets a higher
selected signal as the training-implied long direction.

The stage includes:

- q10 15-minute sleeves;
- q05 30-minute sleeves;
- turnover-controlled q05 30-minute sleeves;
- paired Hawkes gated and shared-sample ungated variants;
- configured one-way cost scenarios.

By default, portfolio dates at or before the selection freeze date are skipped.
Setting `allow_in_sample_portfolio: true` is diagnostic only and should not be
used for deployment conclusions.

## Parallelism and memory

Each stage has its own maximum worker count and estimated per-worker memory.
Admission is based on current available memory minus a fixed reserve. When
headroom is insufficient, new workers are not started; running workers are
allowed to checkpoint or finish.

Default starting points for a 128 GB machine:

| Stage | Max date workers | Estimated GB/worker |
|---|---:|---:|
| materialize | 6 | 12 |
| basic_screen | 8 | 6 |
| detailed | 4 | 16 |
| portfolio | 6 | 8 |

These are admission limits, not promises that every stage will reach its cap.
Use the stage status files and real RSS measurements to recalibrate.

## Commands

Complete staged run:

```powershell
python nff_research/v2_8_launch.py `
  --config configs/v2_8_pipelined_full_campaign.yaml `
  --stage all
```

Run or resume one stage:

```powershell
python nff_research/v2_8_launch.py `
  --config configs/v2_8_pipelined_full_campaign.yaml `
  --stage materialize

python nff_research/v2_8_launch.py `
  --config configs/v2_8_pipelined_full_campaign.yaml `
  --stage basic_screen

python nff_research/v2_8_launch.py `
  --config configs/v2_8_pipelined_full_campaign.yaml `
  --stage select

python nff_research/v2_8_launch.py `
  --config configs/v2_8_pipelined_full_campaign.yaml `
  --stage detailed

python nff_research/v2_8_launch.py `
  --config configs/v2_8_pipelined_full_campaign.yaml `
  --stage portfolio
```

Every date-stage has its own `_SUCCESS`, logs and retry state. Re-running a
stage reuses completed date-stage outputs with the same contract.

## Anti-leakage rules

1. All 464 factors may be raw-screened, but selection uses only training dates.
2. Factor sign is frozen from the training-window mean rank IC.
3. Portfolio dates begin strictly after `selection_end_date` by default.
4. Hawkes gates do not select the factor direction; they only alter eligibility.
5. Detailed in-sample diagnostics may be generated, but portfolio conclusions
   must be based on post-freeze dates.

## Remaining validation

CI validates imports, candidate quotas, stage contracts, direction handling and
memory-admission guards. A real one-date-per-stage benchmark and an interrupted
resume test against the local NFF warehouse are still required before marking
the branch production-ready.
