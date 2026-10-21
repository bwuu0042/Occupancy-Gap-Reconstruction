# Performance summary

Formatted from the output of `scripts/profile_methods.py` (full printout: [`run_output.txt`](run_output.txt), cProfile: [`cprofile.txt`](cprofile.txt)).

**How to read this.** Accuracy is MAE in people on busy G.20 test gaps (lower is better). Times are per gap on a shared Colab CPU and shift between runs (scoring 200 gaps at each length took 19 s in one run and 35 s in another), so read them as orders of magnitude. Operation counts are hand estimates in which one multiply, add, compare or exponential counts as one operation.

## 1. Accuracy, cost and interpretability

| Method | MAE 1 h | MAE 24 h | Time per gap, 24 h (ms) | Time per gap, mean of 8 lengths (ms) | Fit time | Ops per 24 h gap | Interpretability |
|---|---:|---:|---:|---:|---:|---:|---|
| carry_forward | 1.563 | 1.426 | 0.006 | 0.006 | none | 0 | very high |
| linear_interp | **1.230** | 1.401 | 0.016 | 0.017 | none | 576 | very high |
| weekly_profile | 1.773 | 0.968 | 0.003 | 0.003 | none | 0 | high |
| decay_profile | 1.235 | 0.931 | 0.045 | 0.039 | none | 1,728 | high (one parameter, tau) |
| tree_sq_d5 | 1.428 | 1.037 | 0.075 | 0.059 | 0.83 s | 4,896 | medium (readable splits) |
| tree_abs_d5 | 1.411 | **0.829** | 0.074 | 0.058 | 91.27 s | 4,896 | medium (readable splits) |
| svr_c003 | 1.506 | 0.837 | 76.995 | 20.513 | 1.81 s | 32,872,896 | low (kernel model) |

Bold marks the lowest MAE in each accuracy column. Interpretability is a qualitative judgement, not a measurement.

## 2. Model fitting

| Model | Training rows | Fit time (s) |
|---|---:|---:|
| tree_sq_d5 | 239,264 | 0.83 |
| tree_abs_d5 | 60,000 | 91.27 |
| svr_c003 | 8,000 | 1.81 |

The training set itself (239,264 rows from 4,000 simulated gaps) builds in 0.30 s. The rule-based methods need no fitting.

## 3. Prediction time by gap length (ms per gap)

| Gap | carry_forward | linear_interp | weekly_profile | decay_profile | tree_sq_d5 | tree_abs_d5 | svr_c003 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 min | 0.005 | 0.015 | 0.002 | 0.034 | 0.047 | 0.046 | 0.775 |
| 30 min | 0.005 | 0.016 | 0.002 | 0.033 | 0.062 | 0.061 | 1.384 |
| 1 h | 0.005 | 0.015 | 0.002 | 0.033 | 0.045 | 0.044 | 2.736 |
| 2 h | 0.005 | 0.015 | 0.002 | 0.033 | 0.047 | 0.046 | 5.512 |
| 4 h | 0.005 | 0.015 | 0.002 | 0.035 | 0.047 | 0.046 | 14.629 |
| 8 h | 0.010 | 0.025 | 0.004 | 0.056 | 0.079 | 0.078 | 29.461 |
| 12 h | 0.005 | 0.017 | 0.003 | 0.040 | 0.071 | 0.070 | 32.614 |
| 24 h | 0.006 | 0.016 | 0.003 | 0.045 | 0.075 | 0.074 | 76.995 |

The rule-based methods and trees stay roughly flat with gap length. The SVR grows about linearly, because every hidden bin is evaluated against 4,389 support vectors.

## 4. Operation counts (hand estimates)

| Method | Ops per bin | Ops per 24 h gap (288 bins) | Note |
|---|---:|---:|---|
| carry_forward | 0 | 0 | |
| linear_interp | 2 | 576 | |
| weekly_profile | 0 | 0 | lookup only |
| decay_profile | 6 | 1,728 | |
| tree_sq_d5 | 17 | 4,896 | |
| tree_abs_d5 | 17 | 4,896 | |
| svr_c003 | 114,142 | 32,872,896 | 4,389 support vectors |

The SVR does about 19,000 times the arithmetic of the decay rule but takes about 1,700 times as long, so the rule-based methods are limited by call overhead rather than arithmetic.

## 5. Memory

| Stage | Peak (MB) |
|---|---:|
| Series build (one room) | 386.6 |
| Training set (4,000 gaps) | 35.5 |
| Scoring (50 gaps x 8 lengths) | 3.2 |
| Whole process, peak RSS | 1,033 |

The first three rows are Python and numpy allocations measured with `tracemalloc`. Peak RSS also includes the loaded event table and memory allocated inside C libraries such as libsvm, which `tracemalloc` cannot see.
