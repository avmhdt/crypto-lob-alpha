# Signal model artifact

Contract v1.0.0, the producer's copy. market-making-sim mirrors it as `docs/signal-artifact.md`. The spec's section 10 holds the normative text. This page restates it and defines every feature precisely enough to recompute it from the canonical `book_l2` and `trades` streams.

## Where it lives

```
artifacts/models/<model_name>/<model_version>/
    model.txt            LightGBM booster, text format
    feature_spec.json    features, parameters, normalisation (docs/feature_spec.schema.json)
    metadata.json        provenance, fold dates, SHA-256 values, metrics
    parity_sample.csv    >= 1,000 consecutive rows of one test day: raw and normalised features and predictions
```

- `model_name` is `lob-<venue>-<symbol>-<target>`, for example `lob-binance-usdm-BTCUSDT-dmid_ticks_time_1000`.
- `model_version` is semver. Each walk-forward fold export gets its own version. A new fold or a retrain bumps the minor version, and a changed feature set or target bumps the major version.
- A pre-release such as `0.0.0-pre.20261126` is provisional. Its `metadata.json` has `"provisional": true` and its `status` says so. It exists so the consumer can test against a real model; no published result may use it.
- A version folder is immutable once written: `lob export-model` refuses to write into an existing one.

Command: `lob export-model --run-id <id> --version <semver> [--fold k] [--provisional]` exports one fold of a LightGBM `lob train` run. The default fold is the last.

## Loading and checks

`lob_alpha.export.load_model(folder)` and `tests/contract/test_artifact_contract.py` (which needs only `lightgbm`, `json`, `csv` and `hashlib`, so it can be copied to the consumer) check:

1. `spec_version` major is 1.
2. The SHA-256 of `model.txt` equals `metadata.model_sha256`, and those of `feature_spec.json` and `parity_sample.csv` equal `metadata.files_sha256`.
3. The LightGBM major.minor equals `metadata.lightgbm_version`. The current artifacts are written with LightGBM 4.7.
4. The booster's feature order equals the order of `feature_spec.json` `features[].name`.
5. The parity sample's predictions are reproduced from its `__norm` columns within 1e-9 absolute, and each `__norm` value is reproduced from its raw column by the normalisation.

## metadata.json

| Key | Meaning |
|---|---|
| `git_sha` | Commit that exported the artifact |
| `created_utc` | Export time |
| `lightgbm_version` | LightGBM that wrote `model.txt` |
| `model_sha256`, `files_sha256` | SHA-256 of the other three files |
| `fold` | Walk-forward fold index of the run |
| `train_start`, `train_end`, `train_days` | Training days (UTC dates; complete days, so the list can skip dates) |
| `valid_start`, `valid_end`, `validation_days` | Validation day (early stopping) |
| `test_days` | Days the fold was evaluated on; strictly after `valid_end` |
| `target_unit` | `ticks` |
| `metrics` | `oos_r2`, `ic_spearman` and `n_test_days`, on `test_days` only |
| `run_id` | The `lob train` run, a row in `results/trial_log.csv` |
| `provisional`, `status` | Present on pre-releases |

## The grid and the target

- **Clock `time`:** one row every 100 ms at UTC multiples of 100 ms, only while the book is LIVE: no logged gap, and not after the last recorded batch.
- **What a row sees:** the row at `t` reflects every book batch and trade whose known-at time is `<= t`. Known-at time is `ts_recv` made non-decreasing, as a running maximum, because the host clock steps back about 1 ms every 30 s.
- **Target:** `dmid_ticks_time_1000` = mid(t + 1000 ms) − mid(t), in ticks, where mid is (best bid + best ask) / 2 in ticks. It is null across gaps, book resets and 00:00 UTC.
- **Prediction:** the expected value of that change, in ticks.

## Windows

- **Time windows** `(t − W, t]` cover the book batches or trades whose known-at time falls in that interval.
- **Event windows** are the last N book batches (unit `events`) or the last N trades (unit `trades`) known at `t`.
- **Resets:** windows never reach across a book reset (the snapshot batch that starts a new sync epoch) or across 00:00 UTC. Near a reset, the sum covers only the part of the window since the reset.
- **Consumers:** an engine that warms up from its first event agrees with these values once every window is filled, and once the last reset and the last 00:00 UTC are at least the longest window in the past. The parity rows are chosen to satisfy both conditions: they start at 12:00 UTC or later and contain no gap.

## Normalisation

The model input for feature `x` is `clip((x − mean) / std, −5, 5)`, using the `normalization` values in `feature_spec.json`. They are fitted on the training days only. A missing raw value stays missing (NaN), which LightGBM treats as missing. Every training statistic, including `depth_norm`, the integrated-OFI weights and the micro-price table, comes from the training days alone (AC-W-05).

## Feature kinds, with a worked example

The example uses two consecutive batches, n − 1 and n, with prices in ticks and sizes in base units:

| | bid_1 | bid_qty_1 | ask_1 | ask_qty_1 |
|---|---|---|---|---|
| batch n − 1 | 1000 | 2.0 | 1001 | 3.0 |
| batch n | 1000 | 2.5 | 1001 | 1.0 |

`depth_norm` = 12.5 in this example, which is the training mean of (bid_qty_1 + ask_qty_1) / 2.

| Kind | Definition | Example |
|---|---|---|
| `ofi` | **Cont, Kukanov and Stoikov (2014) e_n at level 1**, summed over the window and divided by `params.depth_norm`: e_n = 1[Pb_n ≥ Pb_{n−1}] qb_n − 1[Pb_n ≤ Pb_{n−1}] qb_{n−1} − 1[Pa_n ≤ Pa_{n−1}] qa_n + 1[Pa_n ≥ Pa_{n−1}] qa_{n−1}. e_n is 0 for the first batch after a reset. | e_n = 2.5 − 2.0 − 1.0 + 3.0 = 2.5. A window holding only this batch: 2.5 / 12.5 = 0.2 |
| `mlofi` | **The same e_n with level-m prices and sizes**, m = `levels[0]`, summed and divided by `depth_norm`. A missing level has price −1 and size 0. | Same arithmetic, using the level-m columns |
| `integrated_ofi` | **Σ_l `params.weights[l−1]` × (level-l OFI sum over the window)**, divided by `depth_norm`, for l = 1..10. The weights are the first principal component of the depth-normalised level 1..10 OFI on the training rows (covariance PCA), scaled so that Σ\|w\| = 1 and Σ w > 0 | With weights (0.5, 0.3, 0.2, 0, …) and level sums (2.5, 1.0, −0.5, …): (1.25 + 0.3 − 0.1) / 12.5 = 0.116 |
| `queue_imbalance` | **(Σ bid_qty_1..L − Σ ask_qty_1..L) / (Σ bid_qty_1..L + Σ ask_qty_1..L)**, where L = max(`levels`). It is missing when both sums are 0 | L = 1 at batch n: (2.5 − 1.0) / 3.5 = 0.428571… |
| `microprice` | **Stoikov (2018) g(I, S) in ticks**, with I = bid_qty_1 / (bid_qty_1 + ask_qty_1) and S = ask_1 − bid_1 clamped to 1..`params.max_spread_ticks`. `params.table` rows are `[S, imbalance_lower_edge, g]`, with `imbalance_buckets` equal buckets. Take the last row for spread S whose edge is ≤ I (I = 1 falls in the last bucket). The table is fitted on consecutive training grid rows, symmetrised, and drops mid moves larger than one tick | I = 2.5 / 3.5 = 0.714. With 10 buckets that is the row `[1, 0.7, g]`, so the value is that row's g |
| `weighted_mid` | **(bid_1 · ask_qty_1 + ask_1 · bid_qty_1) / (bid_qty_1 + ask_qty_1) − (bid_1 + ask_1) / 2**, in ticks, which equals (I − ½) · S | (0.714286 − 0.5) · 1 = 0.214286 |
| `spread` | **ask_1 − bid_1**, in ticks | 1 |
| `trade_sign_imbalance` | **Σ side · w / Σ w over the window**, with side +1 for buyer-initiated and −1 for seller-initiated (DQ-06). `params.weight` is `volume` (w = trade size) or `count` (w = 1) | Trades (+1, 0.5), (−1, 0.2), (+1, 0.1): volume (0.5 − 0.2 + 0.1) / 0.8 = 0.5; count (1 − 1 + 1) / 3 = 0.3333 |
| `mid_changes` | **Number of book batches in the window whose mid differs from the previous batch's mid.** The first batch after a reset does not count | Mids 1000.5, 1000.5, 1001, 1000.5: 2 |
| `realized_vol` | **sqrt(Σ (mid_k − mid_{k−1})²)** over the 100 ms grid points k in the window, in ticks. A change is counted at the later point, and no change is counted at the first grid point of a LIVE stretch | W = 300 ms, grid mids at t − 300, t − 200, t − 100 and t of 1000.5, 1001, 1001 and 1000.5: changes into the last three points are 0.5, 0 and −0.5, so sqrt(0.5) = 0.7071 |

## parity_sample.csv

- **Columns:** `ts_exchange`, `ts_recv`, `symbol`, then each feature by name (raw, as defined above, after `depth_norm`), then each normalised feature as `<name>__norm`, then `prediction`.
- **Format:** UTF-8, a header row, LF line endings, floats written with 17 significant digits.
- **Timestamps:** `ts_recv` is the grid time t. `ts_exchange` is the exchange time of the last book batch known at t.
- **Rows:** 1,200 consecutive 100 ms grid rows of the fold's test day, starting at the first clean window at or after 12:00 UTC. A clean window has every feature present and no missing grid point.
