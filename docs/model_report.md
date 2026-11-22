# Occupancy model – evaluation

> ⚠️ **Trained on SYNTHETIC data** – for development only, not real results.

- Target: free spots **30 min** ahead, per garage
- Data: synthetic (2026-10-04 – 2026-11-22, 154,201 rows, 34 garages)
- Test set: last 20% of the period (time-based split)
- Best model: **Random Forest** (refit on all data and saved)
- Trained: 2026-11-22T11:51:28+00:00

## Metrics on the test set (in free spots, lower is better)

| Model | MAE | RMSE | MAE vs. baseline |
|---|---:|---:|---:|
| Random Forest | 7.03 | 15.70 | -71% |
| Gradient Boosting | 7.35 | 14.22 | -70% |
| Baseline: stays the same | 10.16 | 22.90 | -58% |
| Baseline: same time last week | 24.45 | 78.62 | +0% |

## Feature importance

Permutation importance on the test set: how much larger the MAE gets (in free spots)
when a feature's values are shuffled, i.e. when the model can no longer use it.
Features that do not vary in the test period (e.g. no holiday in it) show 0.

![Feature importance](feature_importance.svg)

| Feature | Extra MAE when shuffled |
|---|---:|
| Free spots now | +204 |
| Free spots 15 min ago | +42 |
| Mean of last hour | +15 |
| Same time last week | +10 |
| Free spots 1 h ago | +8.9 |
| Occupancy rate now | +2.8 |
| Hour of day | +0.90 |
| Capacity | +0.52 |
| Garage | +0.08 |
| Weekday | +0.02 |
| Weekend | +0.01 |
| Public holiday | +0.00 |
| Month | +0.00 |
