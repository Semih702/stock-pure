# TimesNet (stockpure) — accuracy & timing
- dataset: hf_bitcoin (resample=1H)
- context=96, horizon=24, epochs=3, device=cpu
- train_time_s: 2.543
- infer_time_ms: mean=7.164, std=1.125

| model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| stockpure | 20657.8359 | 20796.7441 | 0.9985 |
| naive | 20689.2285 | 20829.9629 | 0.9999 |
| seasonal_naive | 20689.2988 | 20830.0312 | 0.9999 |
