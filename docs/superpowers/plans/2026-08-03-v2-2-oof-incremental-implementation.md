# NFF v2.2 OOF Incremental Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-sample bundle ridge diagnostic with reproducible fixed-symbol-fold OOF validation and finish the v2.2 portfolio, reporting, versioning, and CI contracts.

**Architecture:** Keep the existing daily runner and add small pure helpers for stable folds, train-only ridge preprocessing, OOF prediction, and OOF metrics. The daily screen builds one largest-step common sample per minute and admits a minute only when every step completes all folds. Portfolio output receives deterministic gate pairing, applicability, capacity proxies, and primary/diagnostic execution aggregates.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, GitHub Actions.

---

### Task 1: OOF Fold and Ridge Primitives

**Files:**
- Modify: `nff_research/v2_1_neutralized_runner.py`
- Create: `tests/test_nff_research_v2_2.py`

- [ ] Add `test_symbol_fold_known_vectors` for normalization, UTF-8 encoding, first-eight-byte unsigned big-endian SHA256 conversion, and modulo five. Assert lowercase/whitespace symbols map identically.
- [ ] Run `python -m pytest -q tests/test_nff_research_v2_2.py::test_symbol_fold_known_vectors --tb=short`; expect failure because `stable_symbol_fold` is absent.
- [ ] Implement `stable_symbol_fold(symbol, folds=5)` with the exact approved hash contract.
- [ ] Run the same node; expect PASS.
- [ ] Add `test_oof_ridge_heldout_label_perturbation_does_not_change_heldout_prediction`. The fixture calls the wished-for OOF helper, perturbs only fold-0 labels, and asserts fold-0 predictions are unchanged while at least one other fold may change. Also assert feature ranks use `method="average", pct=True`, training labels are demeaned, and scaling/intercept statistics are learned from training rows only.
- [ ] Add `test_oof_high_dimensional_noise_does_not_reproduce_insample_overfit` with a seeded `n=80, p=70` noise fixture; assert in-sample RankIC is above `0.90` and OOF absolute RankIC is below `0.20`.
- [ ] Add `test_oof_subprocess_does_not_retain_minute_arrays`. A subprocess repeatedly evaluates seeded 1,000-row/55-feature blocks, forces GC, and asserts final RSS growth remains below 128 MB; this guards incremental OOF retention, not full-worker memory.
- [ ] Run `python -m pytest -q tests/test_nff_research_v2_2.py -k "heldout_label or high_dimensional or subprocess" --tb=short`; expect failures because the OOF helper is absent.
- [ ] Split the old ridge helper into train-only `_ridge_fit` and `_ridge_predict`, then implement `_fixed_symbol_fold_oof_predict` with defaults `folds=5`, `alpha=0.001`, and `min_train_n=40`.
- [ ] Run all Task 1 tests; expect PASS.
- [ ] Run the complete Task 1 test selection and verify deterministic PASS on two consecutive runs.
- [ ] Run `git add -- nff_research/v2_1_neutralized_runner.py tests/test_nff_research_v2_2.py`.
- [ ] Commit with `git commit -m "Add fixed-symbol-fold OOF ridge primitives"`.

### Task 2: Common-Sample OOF Bundle Screen

**Files:**
- Modify: `nff_research/v2_1_neutralized_runner.py`
- Modify: `tests/test_nff_research_v2_2.py`

- [ ] Add `test_pooled_prediction_ic_has_no_cross_section_size_bias` using two fixtures with identical within-minute ranks but different cross-section-size configurations. Assert both equal the same hand-calculated per-minute percentile-rank/demean correlation and equal each other.
- [ ] Add `test_oof_metric_formulas_and_delta_signs` with hand-calculated prediction/target vectors. Assert exact minute RankIC, pooled IC, R-squared, MSE, `new - previous` IC/R-squared deltas, `previous - new` MSE improvement, and incremental-prediction correlation.
- [ ] Add `test_bundle_oof_screen_uses_identical_common_sample_indices_minutes_and_folds` with four bundle steps and intentionally staggered missing values; assert every emitted step reports identical admitted minute identities, sample/fold identity hashes, `fold_count == 5`, `common_feature_count`, `sample_count`, and `sampled_minutes`.
- [ ] Add `test_bundle_oof_screen_discards_minute_when_any_step_fails` against `bundle_incremental_model_screen`; monkeypatch one bundle step's OOF call to return a NaN prediction for one minute and assert that minute is absent from every emitted step.
- [ ] Run `python -m pytest -q tests/test_nff_research_v2_2.py -k "pooled_prediction or metric_formulas or bundle_oof" --tb=short`; expect failures from absent helpers and the current in-sample screen.
- [ ] Implement a pure pooled demeaned percentile-rank helper and exact OOF metric helper.
- [ ] Replace `bundle_incremental_model_screen` internals with largest-step common complete cases, shared feature ranks/folds, all-step minute admission, and OOF predictions. Emit `sample_count`, `sampled_minutes`, fold count, common feature count, sample/fold identity hashes, OOF IC/R-squared/MSE, positive-is-better deltas, and incremental-prediction correlation.
- [ ] Rename daily output to `bundle_oof_incremental_screen.*` and make contract text explicitly cross-sectional OOF, not temporal OOS.
- [ ] Run the Task 2 test selection; expect PASS.
- [ ] Run `git add -- nff_research/v2_1_neutralized_runner.py tests/test_nff_research_v2_2.py docs/superpowers/plans/2026-08-03-v2-2-oof-incremental-implementation.md`.
- [ ] Commit with `git commit -m "Replace bundle diagnostics with common-sample OOF validation"`.

### Task 3: Portfolio Applicability and Controlled Hawkes Gate

**Files:**
- Modify: `nff_research/v2_1_neutralized_runner.py`
- Modify: `tests/test_nff_research_v2_2.py`

- [ ] Add `test_contrarian_q05_weights_reverse_signal_and_select_exact_tails`, `test_same_sleeve_turnover_ignores_other_active_sleeves`, and `test_portfolio_applicability_contract` for the approved horizon/rebalance map.
- [ ] Add `test_hawkes_gate_pair_uses_shared_sample_and_deterministic_top_20_percent`. Include intensity ties and missing values; assert the paired ungated/gated inputs share the finite pre-gate index and the gated index removes exactly `ceil(0.2*n)` rows by intensity-descending/symbol-ascending order.
- [ ] Add `test_vwap_portfolio_is_primary_and_emits_gate_capacity_fields`. Assert columns `execution_role`, `variant_applicability`, `gate_pair_id`, `gate_mode`, `pre_gate_count`, `eligible_count`, `selected_count`, `gate_kept_ratio`, `pre_gate_adv20_sum`, `eligible_adv20_sum`, `selected_adv20_sum`, and `eligible_adv20_kept_ratio`.
- [ ] Run `python -m pytest -q tests/test_nff_research_v2_2.py -k "contrarian or sleeve or applicability or hawkes_gate or vwap_portfolio" --tb=short`; expect missing-field/behavior failures.
- [ ] Add pure variant-applicability and deterministic gate helpers. Add a paired ungated q05/30m comparator that requires the same finite Hawkes/PIT-ADV20 sample as the gated variant.
- [ ] Update portfolio rows with the exact gate/capacity columns; recover dollar ADV20 as `expm1(control__log_adv20)` and label it as a proxy.
- [ ] Run the Task 3 tests; expect PASS.
- [ ] Run `git add -- nff_research/v2_1_neutralized_runner.py tests/test_nff_research_v2_2.py`.
- [ ] Commit with `git commit -m "Add controlled Hawkes gate and portfolio applicability"`.

### Task 4: v2.2 Contracts and Aggregate Outputs

**Files:**
- Modify: `nff_research/v2_1_neutralized_runner.py`
- Modify: `configs/v2_1_neutralized_full.yaml`
- Modify: `nff_research/README.md`
- Modify: `tests/test_nff_research_v2_2.py`

- [ ] Add `test_v2_2_effective_and_run_contract_hash_all_oof_policies` for `research_version`, `base_runner_version`, fold count/algorithm, ridge alpha, minimum training sample, common-sample policy, applicability policy, primary execution label, and hash change when any field changes.
- [ ] Add `test_v2_2_rejects_existing_v2_1_output_root` by writing a mismatched run contract and asserting a hard `RuntimeError`.
- [ ] Add `test_aggregate_emits_v2_2_primary_diagnostic_and_gate_pair_outputs`. Assert exact artifacts `research_contract_v2_2.json`, `final_report_v2_2.md`, `staggered_portfolio_proxy_primary_vwap_overall.csv/parquet`, `staggered_portfolio_proxy_diagnostic_open_overall.csv/parquet`, and `hawkes_gate_pair_comparison_overall.csv/parquet`. Assert gate-pair deltas `delta_gross_return`, `delta_turnover`, `delta_cost`, `delta_net_return`, `delta_selected_count`, and `delta_selected_adv20_sum`, plus all approved grouping keys.
- [ ] Add `test_aggregate_hac_uses_daily_oof_metric_and_delta_series_directly` with known daily OOF metrics and daily deltas; compare emitted HAC values to direct `_hac_tstat` calls on each exact series and assert they are not obtained by subtracting independently aggregated step statistics.
- [ ] Run `python -m pytest -q tests/test_nff_research_v2_2.py -k "contract or aggregate" --tb=short`; expect v2.1 names and missing artifacts to fail.
- [ ] Add CLI/config propagation for `--oof-folds 5`, `--oof-ridge-alpha 0.001`, and `--oof-min-train-n 40`; include every approved policy in effective/run contracts and hashes.
- [ ] Aggregate `bundle_oof_incremental_screen` daily metrics/deltas with HAC inference. Emit the exact primary, diagnostic, and gate-pair artifacts and v2.2 report/contract filenames.
- [ ] Update config run name to a new v2.2 OOF root, add `research_version: "2.2"` and `base_runner_version: "2.1"`, and update README terminology.
- [ ] Run the Task 4 tests; expect PASS.
- [ ] Run `git add -- nff_research/v2_1_neutralized_runner.py configs/v2_1_neutralized_full.yaml nff_research/README.md tests/test_nff_research_v2_2.py`.
- [ ] Commit with `git commit -m "Version v2.2 OOF contracts and aggregate outputs"`.

### Task 5: CI and Verification

**Files:**
- Modify: `.github/workflows/nff-adapter.yml`
- Modify: `tests/test_nff_research_v2_2.py`

- [ ] Add `test_resource_snapshot_reports_worker_rss` asserting `worker_process_count`, `worker_rss_total_gb`, and `worker_rss_max_gb` are present.
- [ ] Run `python -m pytest -q tests/test_nff_research_v2_2.py::test_resource_snapshot_reports_worker_rss --tb=short`; verify RED because the fields are absent.
- [ ] Update `resource_snapshot` to emit the three worker-RSS fields from live child processes.
- [ ] Rerun the same node; expect PASS.
- [ ] Run `python -m pytest -q tests/test_nff_research_v2_2.py::test_oof_subprocess_does_not_retain_minute_arrays --tb=short` twice; expect deterministic PASS.
- [ ] Add the v2.2 research tests to Windows/Linux CI.
- [ ] Run `C:\nff_envs\qlib-nff-py311\Scripts\python.exe -m pytest -q tests/test_nff_research_v2_2.py tests/test_nff_adapter.py --tb=short`.
- [ ] Run `python -m py_compile nff_research/v2_1_neutralized_runner.py nff_research/v2_multilabel_runner.py tests/test_nff_research_v2_2.py`.
- [ ] Run `git diff --check` and inspect the full diff against the approved design.
- [ ] Run `git add -- .github/workflows/nff-adapter.yml nff_research/v2_1_neutralized_runner.py tests/test_nff_research_v2_2.py`.
- [ ] Commit with `git commit -m "Test v2.2 OOF research contracts in CI"`.
- [ ] Push `agent/nff-direct-qlib-adapter` as explicitly requested by the user for code review; do not launch the formal research run.

### Deferred Run Acceptance

After code review approval, run the same representative date set `2026-06-01..2026-06-16` at worker ladders 1, 4, 8, and 12. For each ladder use a unique output root such as `D:\DEV\AnotherNetworkFactory\warehouses\NFF_research\benchmarks\v2_2_oof_parallel_<N>` and run:

```powershell
C:\nff_envs\qlib-nff-py311\Scripts\python.exe nff_research\v2_1_neutralized_runner.py --config configs\v2_1_neutralized_full.yaml --start-date 2026-06-01 --end-date 2026-06-16 --parallel <N> --max-parallel <N> --min-parallel <N> --out-root D:\DEV\AnotherNetworkFactory\warehouses\NFF_research\benchmarks\v2_2_oof_parallel_<N>
```

Use each root's `resource_samples.ndjson` fields `worker_process_count`, `worker_rss_total_gb`, and `worker_rss_max_gb`, plus `status.json`, per-date `_SUCCESS`, and SHA256 of every aggregate parquet to write `concurrency_benchmark_summary.csv`. Select the lowest `<N>` whose dates/hour is within 5% of peak observed throughput while every sample remains below 78% system memory use, above 32 GB available, and without rising worker RSS after completed-date count stabilizes. Any checksum or row-count mismatch rejects that ladder.
