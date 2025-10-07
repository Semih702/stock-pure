# TimesNet (upstream) — accuracy & timing
- dataset: hf_bitcoin (resample=1H)
- context=96, horizon=24, epochs=3, device=cpu
- train_time_s: 3.535
- infer_time_ms: mean=8.891, std=1.631

| model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| upstream | 20657.8359 | 20796.7441 | 0.9985 |
| naive | 20689.2285 | 20829.9629 | 0.9999 |
| seasonal_naive | 20689.2988 | 20830.0312 |
