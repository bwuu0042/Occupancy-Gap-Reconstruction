# Occupancy Gap Reconstruction

When a room's occupancy sensor goes silent, how well can the missing headcount be filled in? This project hides known stretches of data in two rooms, **G.20 and G.25 in Building 60**, rebuilds them with seven methods, and measures the error with confidence intervals.

> Reconstructed values are estimates, not observed counts. "Truth" throughout is the sensor's own recorded headcount, not an independent count of people.

## At a glance

| | |
|---|---|
| **Data** | Occupancy event log, 3 Mar 2024 to 22 Apr 2026, binned to 5 minutes (224,745 bins; 0.17% unknown in G.20, 0.58% in G.25) |
| **Gap lengths** | 15 min, 30 min, 1 h, 2 h, 4 h, 8 h, 12 h, 24 h |
| **Methods** | Carry-forward, linear interpolation, weekly profile, decay profile, two decision trees (squared and absolute error), SVR |
| **Split** | Chronological 60% train / 20% validation / 20% test, with no shuffling |
| **Test gaps** | 2,000 per gap length in G.20 and 1,000 in G.25, drawn from the test period (seed 9001) |
| **Metrics** | MAE, RMSE and bias on "busy" gaps (gaps that contained occupancy) |
| **Tests** | 17 automated tests, all passing |

## How it works

1. **Build the series.** `scripts/build_data.py` filters the 9-million-row occupancy file to G.20 and G.25, de-duplicates events, and resamples to a regular 5-minute grid. An empty room is held until the next event. An occupied state is held for at most 30 minutes, then marked unknown, so real outages never become ground truth.
2. **Learn a weekly profile** (typical headcount per weekday and half-hour slot) from the training period only.
3. **Hide a gap and reconstruct it.** Every method sees only the two known edge bins, the profile and the gap geometry. The hidden bins are never read (one test checks this).
   - *Decay profile:* the profile plus the edge anomaly (observed minus profile), interpolated across the gap and faded with `exp(-d/τ)`, where `d` is the distance to the nearest known bin. τ = 24 bins (2 hours).
   - *Trees and SVR:* trained on simulated gaps from the training period to predict the anomaly from eight features. Settings: depth 5 with `min_samples_leaf=50`, and SVR with C = 0.03. All were chosen on **G.20 validation gaps only** and applied unchanged to G.25. The SVR trains on 8,000 sampled rows and the absolute-error tree on 60,000, for speed.
4. **Compare with uncertainty.** Each method is compared against a reference (decay profile for G.20, linear interpolation for G.25). Whole days are resampled in a cluster bootstrap, because gaps on the same day are correlated.

## What I found

No method wins everywhere. Best MAE on busy gaps (people):

| Gap | G.20 | G.25 |
| --- | --- | --- |
| 15 min | 1.029, decay profile | 0.501, linear interp. |
| 1 h | 1.230, linear interp. | 0.447, linear interp. |
| 4 h | 1.218, tree (abs. error) | 0.434, linear interp. |
| 12 h | 0.882, SVR | 0.433, tree (abs. error) |
| 24 h | 0.829, tree (abs. error) | 0.437, tree (abs. error) |

- **G.20, short gaps:** the decay profile and linear interpolation are close to tied and clearly beat the trained models.
- **G.20, long gaps:** the SVR and absolute-error tree have the lowest MAE but are biased low (−0.34 to −1.1 people). The decay profile has the lowest RMSE, so the ranking depends on the metric.
- **G.25:** the decay profile's advantage did not carry over. Linear interpolation wins up to about 4 h, and the absolute-error tree is marginally best beyond that. At 24 h the squared-error tree degrades badly (RMSE 1.41).
- **Drift matters.** Mean headcount fell after the training period, so the weekly profile over-predicts the test period by 0.096 people in G.20 and 0.386 in G.25. Profile-based methods are biased upward in G.25 as a result.
- Re-running the G.20 test reproduced the earlier results exactly (maximum MAE difference 0.0).

Full tables are in [`results/final_test_summary_busy_gaps.csv`](results/final_test_summary_busy_gaps.csv) and [`results/confidence_intervals.csv`](results/confidence_intervals.csv). The plots are [`mae_by_gap_length.png`](results/mae_by_gap_length.png) and [`rmse_by_gap_length.png`](results/rmse_by_gap_length.png).

## Reproduce

The analysis was developed in Google Colab. The report notebook clones the repo, mounts Drive and installs the requirements.

```bash
git clone https://github.com/bwuu0042/Occupancy-Gap-Reconstruction.git
cd Occupancy-Gap-Reconstruction
pip install -r requirements.txt
```

**Run the tests and generate the docs:**

```bash
python -m pytest tests -q
python -m pdoc src -o docs
python -m pytest tests -q --html=docs/test_report.html --self-contained-html
```

The tests cover room-key matching, event de-duplication, series resampling, gap eligibility, absence of leakage in every method and in the features, training-period-only profiles, and the cluster bootstrap.

**Rebuild the processed data** (needs the raw files, which are not in this repo):

```bash
python -m scripts.build_data --raw /path/to/raw --out /path/to/processed
```

**Re-run the analysis:** open `notebooks/01_report.ipynb` and run all cells. It expects `headcount_5min_G20_G25.parquet` in your Drive data folder.

**Optional sensitivity check** (bin size and slot width):

```bash
python -m scripts.sensitivity --series /path/to/headcount_5min_G20_G25.parquet --out docs/sensitivity
```

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | `data` (loading and cleaning), `series` (event log to 5-minute grid), `gaps` (simulation and rule-based methods), `features` and `models` (learned methods), `experiment` (scoring), `stats` (bootstrap) |
| `scripts/` | Data build and sensitivity scripts |
| `tests/` | The 17 automated tests |
| `notebooks/` | The report notebook |
| `results/` | Summary CSVs, confidence intervals, figures |
| `docs/` | Generated HTML documentation and test report |

## Limitations

- Only G.20 and G.25 could be matched to sensor metadata, so the results may not generalise to other room types.
- No environmental sensor is located in either room, so environmental data is not used.
- The record is an event log, so silence is ambiguous. The outage counts (21 in G.20, 53 in G.25) are lower bounds.
- Gaps are simulated because real outages are rare. Real outages may be less random.
- Settings were tuned on G.20 only. The decay rule was designed after looking at test-period gaps, so its results may be slightly optimistic.
- Each room has 96 confidence intervals with no multiple-comparison correction.
- Sensor-derived "truth" means these results measure self-consistency, not agreement with an independent count.

The full list is in the final section of the report notebook. Raw and processed data are excluded from Git because redistribution permission has not been confirmed.

## Licence

`[add licence]`. Dataset permissions are separate from the code.
