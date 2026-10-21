# Occupancy Gap Reconstruction

When a room's occupancy sensor goes silent, how well can the missing headcount be filled in? This project hides known stretches of data in two rooms, **G.20 and G.25 in Building 60**, rebuilds them with seven methods, and measures the error with confidence intervals.

> Reconstructed values are estimates, not observed counts. "Truth" throughout is the sensor's own recorded headcount, not an independent count of people.

**Data:** Monash Smart Infrastructure Occupancy and Environmental Data, provided by the unit teaching team. Raw and processed data are not in this repository.

## At a glance

| | |
|---|---|
| **Data** | Occupancy event log, 3 Mar 2024 to 22 Apr 2026, binned to 5 minutes (224,745 bins per room; 0.17% unknown in G.20, 0.58% in G.25) |
| **Gap lengths** | 15 min, 30 min, 1 h, 2 h, 4 h, 8 h, 12 h, 24 h |
| **Methods** | Carry-forward, linear interpolation, weekly profile, decay profile, two decision trees (squared and absolute error), SVR |
| **Split** | Chronological 60% train / 20% validation / 20% test, with no shuffling |
| **Test gaps** | 2,000 per gap length in G.20 and 1,000 in G.25, drawn from the test period (seed 9001) |
| **Metrics** | MAE, RMSE and bias on "busy" gaps (gaps that contained occupancy) |
| **Tests** | 17 automated tests, all passing |

## How it works

1. **Build the series.** `scripts/build_data.py` filters the 9-million-row occupancy file to G.20 and G.25, de-duplicates events, and resamples to a regular 5-minute grid. An empty room is held until the next event. An occupied state is held for at most 30 minutes, then marked unknown, so real outages never become ground truth.
2. **Learn a weekly profile** (typical headcount per weekday and half-hour slot) from the training period only.
3. **Hide a gap and reconstruct it.** Every method sees only the two known edge bins, the profile and the gap geometry. The hidden bins are never read (two tests check this, one for the rule-based methods and one for the feature builder).
   - *Decay profile:* the profile plus the edge anomaly (observed minus profile), interpolated across the gap and faded with `exp(-d/τ)`, where `d` is the distance to the nearest known bin. τ = 24 bins (2 hours).
   - *Trees and SVR:* trained on simulated gaps from the training period to predict the anomaly from eight features. Settings: depth 5 with `min_samples_leaf=50`, and SVR with C = 0.03. All were chosen on **G.20 validation gaps only** and applied unchanged to G.25. The SVR trains on 8,000 sampled rows and the absolute-error tree on 60,000, for speed.
4. **Compare with uncertainty.** Each method is compared against a reference (decay profile for G.20, linear interpolation for G.25). Whole days are resampled in a cluster bootstrap, because gaps on the same day are correlated.

## What I found

No method wins everywhere. Best MAE on busy gaps (people). "≈" and "tie" mean the 95% interval of the difference includes zero.

| Gap | G.20 | G.25 |
| --- | --- | --- |
| 15 min | decay ≈ linear (1.03) | linear (0.50) |
| 1 h | decay ≈ linear (1.23) | linear (0.45) |
| 4 h | tie: decay, linear, SVR, abs. tree (1.22–1.26) | linear (0.43) ≈ abs. tree (0.45) |
| 12 h | SVR (0.88), abs. tree (0.89) | tie: linear, SVR, abs. tree (0.43–0.46) |
| 24 h | abs. tree (0.83), SVR (0.84) | abs. tree (0.44) |

- **G.20, short gaps:** the decay profile and linear interpolation are close to tied and clearly beat the trained models. From about 2 h, decay has a lower RMSE than linear interpolation.
- **G.20, long gaps (8 h and above):** the SVR and absolute-error tree have the lowest MAE, but they are biased low (about −0.3 to −0.5 people). The decay profile is essentially unbiased and has the lowest RMSE, so the ranking depends on the metric. The RMSE differences are small (0.04 to 0.15 people).
- **G.25:** the decay profile's advantage did not carry over. Linear interpolation is best or tied up to about 12 h. The absolute-error tree is clearly better than linear only at 24 h (by 0.06 people), and the SVR has a slightly lower RMSE than linear from 8 h. At 24 h the squared-error tree degrades badly (RMSE 1.41).
- **Drift matters.** Mean headcount fell after the training period, so the weekly profile over-predicts the test period by 0.096 people in G.20 and 0.386 in G.25. Profile-based methods are biased upward in G.25 as a result.
- **G.25 adds no information beyond the weekly rhythm.** Once each room's weekly profile is removed, the two rooms' deviations are essentially uncorrelated (about 0.01), so G.25 was not used as a predictor for G.20.
- Re-running the G.20 test reproduced the earlier results exactly (maximum MAE difference 0.0), and re-running G.25 through `src/` did the same.

Full tables are in [`results/final_test_summary_busy_gaps.csv`](results/final_test_summary_busy_gaps.csv) and [`results/confidence_intervals.csv`](results/confidence_intervals.csv). In the interval file, `diff` is positive when a method is worse than the reference, and `real` is true when the 95% interval excludes zero. The plots are [`mae_by_gap_length.png`](results/mae_by_gap_length.png) and [`rmse_by_gap_length.png`](results/rmse_by_gap_length.png).

### Sensitivity and cost

Checked on G.20 with the rule-based methods only (`scripts/sensitivity.py`; results in `docs/sensitivity/`):

- **Bin size (5, 15, 30, 60 min):** errors rise as bins get coarser, partly because the truth itself gets coarser, so absolute values are not comparable across bin sizes. The method ranking is mostly stable. The exception is a 1 h gap at 60 min bins, which is a single bin.
- **Weekly-slot width (15, 30, 60 min):** changes MAE by at most about 0.005, so the 30-minute choice is not sensitive.
- **Decay τ (validation):** 24 bins was best (MAE 0.776), but τ from 1 h to 16 h changes validation error by only about 0.06 people.
- **Tree depth (validation):** squared-error trees were best at depth 5 and overfit beyond it. Validation MAE was 0.800, against 0.776 for the decay profile.
- **SVR C (validation):** C = 0.01, 0.03 and 0.1 are within 0.01 MAE of each other.
- **Run time:** carry-forward and the weekly profile take about 0.03 ms per gap, linear interpolation 0.07 ms, and the decay profile 0.15 ms. The SVR fits in about 2 s on 8,000 rows, and the absolute-error tree in 80 to 100 s on 60,000 rows. Training sizes differ, so these are not like-for-like.
- **Neural network:** an MLP was tried on validation only and dropped. It was under-trained (it hit the 60-iteration cap), took 29 to 52 s to fit, and was no better than the decay profile.

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

The tests cover room-key matching, event de-duplication, series resampling, gap eligibility, absence of leakage in the rule-based methods and the feature builder, training-period-only profiles, and the cluster bootstrap.

**Rebuild the processed data** (needs the raw files, which are not in this repo):

```bash
python -m scripts.build_data --raw /path/to/raw --out /path/to/processed
```

**Re-run the analysis:** open `notebooks/01_report.ipynb` and run all cells. It expects `headcount_5min_G20_G25.parquet` in your Drive data folder. The first run of the final-test section takes about 15 minutes and then caches its results. The check against an earlier G.20 run is optional and only works if that earlier results file exists.

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
| `notebooks/` | The report notebook (`01_report.ipynb`) and the archived exploration notebook |
| `results/` | Summary CSVs, confidence intervals, figures |
| `docs/` | Generated HTML documentation, test report, and sensitivity tables |

## Limitations

- Only G.20 and G.25 could be matched to sensor metadata (the occupancy file has 5 spaces and only 2 appear in the metadata), so the results may not generalise to other room types.
- No environmental sensor with usable data is located in either room, so environmental data is not used.
- The record is an event log, so silence is ambiguous. Genuine outages (a silence of over an hour that began while the room was occupied) numbered 21 in G.20 and 53 in G.25, and these are lower bounds.
- Gaps are simulated because real outages are rare. Real outages may be less random.
- Settings were tuned on G.20 only. The decay rule was designed after looking at test-period gaps, so its results may be slightly optimistic. The reference methods (decay for G.20, linear for G.25) were chosen after seeing results.
- Each room has 96 confidence intervals with no multiple-comparison correction, so borderline intervals should be treated with caution.
- The test window is only about five months and covers the summer break, so some of the drift may be seasonal. This was not checked against the same months of the previous year.
- One gateway device reports both rooms, so a gateway failure would remove both at once.
- Sensor-derived "truth" means these results measure self-consistency, not agreement with an independent count.

## Use of AI

Claude (Anthropic) was used throughout this project for planning the approach, drafting and debugging code, interpreting results, and reviewing this README. All results were produced by running the code on the dataset, and I checked the outputs. Examples of AI errors that were caught and corrected during the work include a feature-matrix shape bug, an overstated statistical test, a misread of the G.25 results, and a depth-4 versus depth-5 slip when reading the tree validation curve. Full details are in [`docs/ai_use_log.md`](docs/ai_use_log.md).

## Licence

MIT, see [`LICENSE`](LICENSE). Dataset permissions are separate from the code.
