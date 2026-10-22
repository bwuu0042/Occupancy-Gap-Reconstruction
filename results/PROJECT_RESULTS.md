# Project Results and Verification

This file brings the main evaluation, test, performance, and sensitivity findings together in one readable summary. Detailed tables, plots, and generated reports remain linked below so the original outputs can be checked.

**Project:** Occupancy Gap Reconstruction  
**Rooms:** G.20 and G.25, Building 60  
**Data period:** 3 March 2024 to 22 April 2026  
**Time step:** 5 minutes  
**Reference values:** recorded occupancy sensor headcounts (not an independent people count)

## At a glance

- Seven methods were evaluated: carry-forward, linear interpolation, weekly profile, decay profile, squared-error decision tree, absolute-error decision tree, and support vector regression (SVR).
- Gaps ranged from 15 minutes to 24 hours.
- The split was chronological: 60% training, 20% validation, and 20% test.
- Main metrics are MAE, RMSE, and bias, reported in people for simulated gaps that contained occupancy.
- No method is best for every room, gap duration, and metric.
- The regenerated HTML test report shows **29 passed, 0 failed** in 960 ms. This matches the README’s reported test count.

## Test and implementation checks

The automated tests cover core method outputs, data handling, feature construction, invalid inputs, and statistical functions. The documented checks include:

- Hand-calculated expected values for carry-forward and linear interpolation.
- Series construction behaviour, including holding an empty state and expiring an occupied state after the configured 30-minute maximum hold.
- An analytic check of the decay-profile formula, its limiting behaviour, and clipping at zero.
- Leakage checks that change hidden values and verify they do not affect predictions or features.
- A check that validation and test data do not affect the training-only weekly profile.
- Input validation and bootstrap edge cases.

**Recorded report:** [HTML pytest report](../docs/test_report.html)  
**Test code:** [tests folder](../tests/)  
**Test command:** `python -m pytest tests -q`  
The report was regenerated after the test suite completed and records 29 passed, 0 failed in 960 ms.

## Final test-set accuracy

MAE and RMSE are in people. Bias is prediction minus sensor reference: negative values mean under-prediction. The sample count is the number of busy gaps used for that row.

| Room | Gap | Method | Busy gaps | MAE | RMSE | Bias |
|---|---:|---|---:|---:|---:|---:|
| G.20 | 15 min | Decay profile | 622 | 1.029 | 1.660 | -0.025 |
| G.20 | 15 min | Linear interpolation | 622 | 1.034 | 1.677 | 0.015 |
| G.20 | 1 h | Linear interpolation | 785 | 1.230 | 2.254 | -0.056 |
| G.20 | 1 h | Decay profile | 785 | 1.235 | 2.283 | -0.130 |
| G.20 | 4 h | Absolute-error tree | 1,172 | 1.218 | 2.951 | -0.556 |
| G.20 | 4 h | SVR | 1,172 | 1.226 | 3.113 | -0.671 |
| G.20 | 12 h | SVR | 1,676 | 0.882 | 2.476 | -0.486 |
| G.20 | 12 h | Absolute-error tree | 1,676 | 0.893 | 2.461 | -0.375 |
| G.20 | 24 h | Absolute-error tree | 1,851 | 0.829 | 2.359 | -0.340 |
| G.20 | 24 h | Decay profile | 1,851 | 0.931 | 2.262 | 0.028 |
| G.25 | 15 min | Linear interpolation | 446 | 0.502 | 0.768 | -0.105 |
| G.25 | 1 h | Linear interpolation | 711 | 0.447 | 0.733 | -0.022 |
| G.25 | 4 h | Linear interpolation | 904 | 0.434 | 0.738 | -0.002 |
| G.25 | 12 h | Absolute-error tree | 992 | 0.433 | 0.748 | -0.102 |
| G.25 | 24 h | Absolute-error tree | 1,000 | 0.437 | 0.761 | -0.084 |
| G.25 | 24 h | Linear interpolation | 1,000 | 0.499 | 0.851 | 0.000 |
| G.25 | 24 h | Squared-error tree | 1,000 | 0.922 | 1.406 | 0.658 |

The final summary contains all seven methods at all eight gap lengths. The table above highlights representative comparisons rather than replacing the full data.

### Interpretation

- In G.20, decay profile and linear interpolation are effectively tied at 15 minutes and one hour by MAE. For longer gaps, the learned models often achieve lower MAE, but they under-predict more. At 24 hours, the decay profile has slightly higher MAE than the absolute-error tree but lower RMSE and almost zero bias.
- In G.25, linear interpolation is strongest for short gaps and remains competitive at longer durations. At 24 hours, the absolute-error tree has lower MAE and RMSE than linear interpolation. The squared-error tree performs poorly at 24 hours.
- The G.25 test-period mean headcount is reported as 0.377 people versus a training-derived weekly-profile mean of 0.763. This drift helps explain why profile-based methods can overestimate later occupancy.
- “Busy” is defined using the hidden reference values (at least one person somewhere in the gap). It is useful for retrospective analysis but would not be known during a real outage.

### Uncertainty

The confidence-interval file compares each method with a reference using a day-cluster bootstrap. A positive difference means the method is worse than the reference; a negative difference means it is better. An interval that includes zero does not establish a clear difference under this analysis.

Examples from the saved intervals:

- G.20 at 15 minutes: linear interpolation versus decay profile has an MAE difference of 0.0052, with 95% interval [-0.0053, 0.0156]. The interval includes zero.
- G.20 at 24 hours: the absolute-error tree is better than decay profile by 0.1018 MAE, with interval [-0.1348, -0.0693].
- G.25 at 24 hours: the absolute-error tree is better than linear interpolation by 0.0624 MAE, with interval [-0.0864, -0.0367].

There are many comparisons (96 intervals per room) and no multiple-comparison correction, so borderline findings should be treated cautiously.

**Full accuracy files:** [test summary CSV](final_test_summary_busy_gaps.csv) · [confidence intervals CSV](confidence_intervals.csv)  
**Figures:** [MAE by gap length](mae_by_gap_length.png) · [RMSE by gap length](rmse_by_gap_length.png)

## Performance and optimisation

Measurements were made on a shared Google Colab CPU. They vary between runs and should be read as approximate relative costs, not guaranteed timings. The performance study compared prediction accuracy, runtime, fitting time, memory, operation-count estimates, and interpretability.

### Accuracy and prediction cost

| Method | MAE at 1 h | MAE at 24 h | Mean prediction time across 8 lengths | Fit time |
|---|---:|---:|---:|---:|
| Carry-forward | 1.563 | 1.426 | 0.006 ms/gap | None |
| Linear interpolation | 1.230 | 1.401 | 0.017 ms/gap | None |
| Weekly profile | 1.773 | 0.968 | 0.003 ms/gap | None |
| Decay profile | 1.235 | 0.931 | 0.039 ms/gap | None |
| Squared-error tree, depth 5 | 1.428 | 1.037 | 0.059 ms/gap | 0.83 s |
| Absolute-error tree, depth 5 | 1.411 | 0.829 | 0.058 ms/gap | 91.27 s |
| SVR, C = 0.03 | 1.506 | 0.837 | 20.51 ms/gap | 1.81 s |

The absolute-error tree has the lowest MAE in the 24-hour G.20 performance comparison, at a small fraction of SVR prediction cost. Its fitting time is much longer than the squared-error tree’s. The SVR’s per-gap cost increases with gap duration: the recorded run ranges from about 0.78 ms for a 15-minute gap to about 77.0 ms for a 24-hour gap. The other methods remain below about 0.08 ms per gap in that run.

### Memory and operation estimates

- Peak Python/NumPy allocation: 386.6 MB for building one room’s series, 35.5 MB for the 4,000-gap training set, and 3.2 MB for scoring 50 gaps across eight lengths.
- Whole-process peak resident memory was about 1,033 MB, including the loaded event table and native-library allocations not fully measured by `tracemalloc`.
- Hand-estimated operations per 24-hour gap: 576 for linear interpolation, 1,728 for decay profile, 4,896 for either tree, and 32,872,896 for SVR (using 4,389 support vectors). These are scaling estimates, not hardware FLOP measurements.

### Sensitivity findings

- **Bin size:** 5, 15, 30, and 60-minute bins were examined. Errors generally increased as bins became coarser, but absolute error values are not directly comparable because the time resolution also changes the target.
- **Weekly slot width:** 15, 30, and 60-minute slots produced very similar results; the README reports MAE changes no greater than about 0.005 people.
- **Decay time constant:** τ = 24 bins (two hours) was selected on validation data. Settings from one to 16 hours changed validation error by only about 0.06 people.
- **Tree depth:** depth 5 was selected on validation; greater depth increased overfitting for the squared-error tree.
- **SVR C:** C values 0.01, 0.03, and 0.1 had validation MAEs within 0.01.
- **MLP trial:** a neural network was dropped after hitting the 60-iteration limit, taking 29–52 seconds to fit, and not beating the decay profile on validation data.

### Detailed optimisation outputs

- [Performance summary and tables](../docs/performance/run_output.md)
- [Prediction timings by gap length](../docs/performance/timing_per_gap.csv)
- [Model fitting times](../docs/performance/fit_times.csv)
- [Memory measurements](../docs/performance/memory.csv)
- [Hand-estimated operation counts](../docs/performance/flops.csv)
- [Method comparison CSV](../docs/performance/comparison.csv)
- [cProfile output](../docs/performance/cprofile.txt)
- [Full saved run output](../docs/performance/run_output.txt)
- [Bin-size sensitivity data](../docs/sensitivity/bin_size.csv)
- [Weekly-slot-width sensitivity data](../docs/sensitivity/slot_width.csv)

## Limitations and interpretation cautions

1. Sensor-derived headcount is the reference, not an independent count of room occupants. The metrics assess self-consistency with the sensor record.
2. Only two rooms were usable; results may not generalise to other rooms or buildings.
3. Gaps are simulated because genuine outages are uncommon. Actual failures may occur under different conditions.
4. The README notes that the decay method was designed after inspecting test-period gaps and that reference methods were chosen after seeing results. This creates a risk of optimistic comparisons and should be disclosed.
5. The test period is about five months and includes the summer break; seasonal drift was not checked against the same months in another year.
6. The occupancy event log is ambiguous during silence. The 30-minute occupied-state hold is a modelling assumption.
7. Performance measurements vary across shared Colab runs. Memory and operation-count estimates have the limitations described above.
8. The raw and processed dataset is provided by the teaching team and is not included in this public repository. Reproduction of the complete analysis requires access to those data files.

## Reproducing the checks

From the repository root, install the pinned dependencies and run:

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

To regenerate the HTML test report:

```bash
python -m pytest tests -q --html=docs/test_report.html --self-contained-html
```

The complete notebook analysis also requires the processed Parquet dataset in the location described in the [README](../README.md).

## Linked project files

- [Project README](../README.md)
- [Main report notebook](../notebooks/01_report.ipynb)
- [Test source files](../tests/)
- [Reusable project code](../src/)
- [HTML source documentation](../docs/src.html)
- [Generated pytest report](../docs/test_report.html)
