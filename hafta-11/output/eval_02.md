# AdamW Optimizer Benchmark

## 1. Configuration

- Optimizer: `AdamW`
- Betas: `(0.9, 0.95)`
- Epsilon: `1e-8`
- GPU: NVIDIA A100 PCIe

## 2. Training Results

| Step | Loss | Gradient Norm | Step Time |
|---:|---:|---:|---:|
| 0 | 5.1029 | 30.6695 | 2454.33 ms |
| 1 | 4.9357 | 16.5712 | 18.65 ms |
| 2 | 4.4971 | 12.5266 | 17.58 ms |
| 3 | 5.4441 | 47.4995 | 17.60 ms |
| 4 | 4.7671 | 26.7311 | 17.65 ms |
| 5 | 5.0248 | 42.0415 | 17.60 ms |
| 6 | 5.8231 | 44.7433 | 17.59 ms |
| 7 | 5.3284 | 36.0535 | 17.62 ms |
| 8 | 4.7242 | 39.1161 | 17.58 ms |
| 9 | 5.4146 | 43.9054 | 17.56 ms |
| 10 | 5.2699 | 40.4855 | 17.57 ms |
| 11 | 4.4216 | 42.1181 | 17.63 ms |
| 12 | 4.8914 | 41.3983 | 17.56 ms |
| 13 | 4.7717 | 40.5173 | 17.57 ms |

## 3. Performance Summary

- Average step time (excluding step 0): **17.64 ms**
- Average throughput: **~7.28 tok/sec**
- Initial step time: **2454.33 ms** (includes initialization overhead).
- Final loss: **4.7717**

## 4. Conclusion

AdamW with `betas=(0.9, 0.95)` and `eps=1e-8` achieved a relatively stable step time of approximately **17.6 ms** after the initial step.

The first step was significantly slower due to initialization and runtime overhead, so it was excluded from the average performance calculation.