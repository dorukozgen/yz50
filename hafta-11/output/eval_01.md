# A100 PCIe Performance Benchmark

## 1. Baseline (FP32)

- GPU: NVIDIA A100 PCIe
- Precision: FP32
- Average step time: **26.88 ms**

## 2. BF16 Autocast

- Method: `torch.autocast(dtype=torch.bfloat16)`
- Average step time: **32 ms**

BF16 autocast was slower than the FP32 baseline in this experiment.

## 3. PyTorch Compile

- Method: `torch.compile`
- Average step time: **~15.5 ms**

`torch.compile` reduced the step time significantly compared to the FP32 baseline.

## 4. Vocabulary Size: 50,304

- Original vocabulary size: 50,257
- Updated vocabulary size: 50,304
- Average step time: **~15.45 ms**

Increasing the vocabulary size from 50,257 to 50,304 had almost no impact on performance.

## 5. Performance Comparison

| Configuration | Step Time |
|---|---:|
| FP32 Baseline | 26.88 ms |
| BF16 Autocast | 32.00 ms |
| `torch.compile` | ~15.5 ms |
| `torch.compile` + Vocab 50,304 | ~15.45 ms |

## 6. Conclusion

- `torch.compile` improved training performance by approximately **42%** compared to the FP32 baseline.
- BF16 autocast was slower in this particular experiment.
- Increasing vocabulary size from 50,257 to 50,304 had negligible performance impact.
- The best-performing configuration was `torch.compile` with a step time of approximately 15.45 ms.

**Note:** Results are based on the observed training steps and may vary depending on GPU utilization, compilation overhead, and other runtime conditions.