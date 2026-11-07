# Gradient Norm Clipping

## 1. Setup

These results come from a **randomly initialized** model: `model = GPT(GPTConfig())` replaced `GPT.from_pretrained('gpt2')`. The runs in eval_03–eval_06 share the same seed (1337), initialization and data order, and each one turns on one more feature on top of the previous one.

| Setting | Value |
|---|---|
| Model | `GPT(GPTConfig())`: GPT-2 (124M) architecture, random init |
| Total parameters | 124,475,904 |
| Weight init | Normal(0, 0.02); residual projections (`c_proj`) scaled by `(2 * n_layer) ** -0.5` |
| Vocabulary size | 50,304 |
| Dataset | Tiny Shakespeare (`input.txt`): 338,024 tokens, 20 batches per epoch |
| Batch | B = 16, T = 1024 → 16,384 tokens per step (no gradient accumulation) |
| Optimizer | `torch.optim.AdamW(lr=3e-4, betas=(0.9, 0.95), eps=1e-8)`, constant LR |
| Gradient clipping | max norm = 1.0 |
| Precision / speed | `torch.autocast(dtype=torch.bfloat16)`, TF32 matmuls, `torch.compile`, Flash Attention |
| Steps | 50 |
| GPU | NVIDIA A100 PCIe 40GB |

## 2. Change

`torch.nn.utils.clip_grad_norm_` was added after `loss.backward()` and before `optimizer.step()`. It treats the gradients of all parameters as one long vector and computes its L2 norm. If the norm is larger than 1.0, every gradient is scaled down so that the norm becomes exactly 1.0. This stops an unlucky batch with a very large gradient from shocking the model with a huge update.

The function returns the norm **before** clipping. This value is printed at every step.

## 3. Training Log (every step)

| Step | Loss | LR | Grad Norm | dt (ms) | tok/sec |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.9473 | 3.0000e-04 | 28.5686 | 29,382.17 | 558 |
| 1 | 9.3884 | 3.0000e-04 | 6.1843 | 115.13 | 142,313 |
| 2 | 8.9498 | 3.0000e-04 | 2.4922 | 125.81 | 130,224 |
| 3 | 8.7639 | 3.0000e-04 | 2.8483 | 128.04 | 127,960 |
| 4 | 8.7683 | 3.0000e-04 | 10.1681 | 130.18 | 125,861 |
| 5 | 8.4546 | 3.0000e-04 | 2.0524 | 126.44 | 129,579 |
| 6 | 8.3357 | 3.0000e-04 | 2.3927 | 128.05 | 127,950 |
| 7 | 8.0623 | 3.0000e-04 | 1.7255 | 129.43 | 126,585 |
| 8 | 7.7674 | 3.0000e-04 | 1.9885 | 127.39 | 128,617 |
| 9 | 7.5182 | 3.0000e-04 | 1.5291 | 129.33 | 126,680 |
| 10 | 7.3087 | 3.0000e-04 | 1.5207 | 129.41 | 126,602 |
| 11 | 7.1405 | 3.0000e-04 | 1.2154 | 127.64 | 128,365 |
| 12 | 6.9368 | 3.0000e-04 | 1.2147 | 130.17 | 125,861 |
| 13 | 6.9427 | 3.0000e-04 | 1.4865 | 130.25 | 125,785 |
| 14 | 6.9074 | 3.0000e-04 | 1.1141 | 128.44 | 127,561 |
| 15 | 6.7046 | 3.0000e-04 | 1.0004 | 130.51 | 125,543 |
| 16 | 6.6561 | 3.0000e-04 | 1.0244 | 128.51 | 127,487 |
| 17 | 6.5792 | 3.0000e-04 | 0.8851 | 127.19 | 128,817 |
| 18 | 6.5346 | 3.0000e-04 | 1.1736 | 131.31 | 124,773 |
| 19 | 6.3466 | 3.0000e-04 | 0.8502 | 129.03 | 126,977 |
| 20 | 6.4492 | 3.0000e-04 | 1.4922 | 130.22 | 125,819 |
| 21 | 6.1712 | 3.0000e-04 | 1.2995 | 131.18 | 124,894 |
| 22 | 6.2903 | 3.0000e-04 | 3.3001 | 128.92 | 127,084 |
| 23 | 6.1708 | 3.0000e-04 | 1.1161 | 130.10 | 125,932 |
| 24 | 6.1296 | 3.0000e-04 | 1.0313 | 130.07 | 125,968 |
| 25 | 6.3456 | 3.0000e-04 | 1.4378 | 128.73 | 127,273 |
| 26 | 6.4505 | 3.0000e-04 | 1.7578 | 130.10 | 125,934 |
| 27 | 6.3141 | 3.0000e-04 | 1.1974 | 127.49 | 128,512 |
| 28 | 6.2039 | 3.0000e-04 | 1.0857 | 132.79 | 123,380 |
| 29 | 6.1316 | 3.0000e-04 | 1.1693 | 127.36 | 128,638 |
| 30 | 6.0931 | 3.0000e-04 | 0.8012 | 129.10 | 126,906 |
| 31 | 6.0954 | 3.0000e-04 | 0.7132 | 130.47 | 125,576 |
| 32 | 6.0573 | 3.0000e-04 | 0.9950 | 127.10 | 128,904 |
| 33 | 6.2059 | 3.0000e-04 | 1.3299 | 130.63 | 125,424 |
| 34 | 6.2834 | 3.0000e-04 | 0.9611 | 130.48 | 125,569 |
| 35 | 6.1118 | 3.0000e-04 | 1.1153 | 129.86 | 126,164 |
| 36 | 6.0829 | 3.0000e-04 | 1.1470 | 130.44 | 125,608 |
| 37 | 6.0769 | 3.0000e-04 | 1.2326 | 131.65 | 124,447 |
| 38 | 6.0576 | 3.0000e-04 | 1.2514 | 133.34 | 122,877 |
| 39 | 5.8871 | 3.0000e-04 | 0.9010 | 129.98 | 126,054 |
| 40 | 6.1741 | 3.0000e-04 | 1.2302 | 129.47 | 126,546 |
| 41 | 5.9365 | 3.0000e-04 | 1.4054 | 131.41 | 124,676 |
| 42 | 6.0785 | 3.0000e-04 | 1.1340 | 129.47 | 126,550 |
| 43 | 5.8903 | 3.0000e-04 | 0.8072 | 131.06 | 125,009 |
| 44 | 5.8466 | 3.0000e-04 | 1.0879 | 129.20 | 126,810 |
| 45 | 6.0936 | 3.0000e-04 | 1.3425 | 131.73 | 124,377 |
| 46 | 6.1975 | 3.0000e-04 | 0.7270 | 130.08 | 125,953 |
| 47 | 6.0568 | 3.0000e-04 | 0.9566 | 128.36 | 127,641 |
| 48 | 5.9868 | 3.0000e-04 | 0.7502 | 131.75 | 124,359 |
| 49 | 5.9020 | 3.0000e-04 | 0.7254 | 128.16 | 127,837 |

Step 0 includes the `torch.compile` compilation time, so it is left out of the timing averages.

## 4. Summary

| Metric | Value |
|---|---:|
| Initial loss | 10.9473 |
| Final loss (step 49) | 5.9020 |
| Lowest loss | 5.8466 (step 44) |
| Gradient norm at step 0 | 28.5686 |
| Minimum gradient norm | 0.7132 (step 31) |
| Mean gradient norm (steps 1–49) | 1.60 |
| Mean gradient norm (last 10 steps) | 1.02 |
| Steps clipped (norm > 1.0) | 38 / 50 |
| Average step time (steps 1–49) | 129.24 ms |
| Average throughput (steps 1–49) | ~126,822 tok/sec |

## 5. Observations

- The initial loss is **10.9473**, close to ln(50,304) ≈ 10.83, which is the loss of a uniform guess over the vocabulary. A randomly initialized model should start here, so the initialization is correct. (The previous version of this report started from 5.10 because it used the pretrained GPT-2 weights.)
- The gradient norm is large at the very beginning (28.57 at step 0) and falls quickly (6.18 at step 1, 2.49 at step 2). After the first few steps it moves between 1 and 2 and settles near 1.0 by the end. Spikes such as 10.17 at step 4 and 3.30 at step 22 are cut down to 1.0 by clipping instead of turning into large updates.
- Clipping was active on 38 of the 50 steps, and not only at the start. Early on it cuts large norms (28.57, 10.17) down hard. With a constant LR the norm keeps hovering around 1.0–1.5 later on, so 10 of the last 20 steps were still clipped, but only slightly.
- The loss falls from 10.95 to about 5.9 in 50 steps. The bumps at steps 20 and 40 (and 25–26, 45–46) repeat every 20 steps, i.e. once per epoch. Those batches are simply harder, so the noise comes from the data and not from instability.

## 6. Implementation

```python
loss.backward()
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # returns the norm before clipping
optimizer.step()
...
log_str = f"step {i} | loss: {loss_accum.item():.6f} | lr: {optimizer.param_groups[0]['lr']:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
print(log_str)
```

## 7. Notes

- Two fixes were made in `train_gpt2.py` before these runs:
  - The init standard deviation was a typo: `std = 0.2` → `0.02`. With 0.2 the initial loss was ~24.5 instead of ~10.9.
  - Autocast was switched from `float16` to `bfloat16`. Without a `GradScaler`, fp16 underflowed 29% of the gradient entries to zero, and 43% once the loss was divided by 32 for gradient accumulation (cosine similarity to the fp32 gradient fell to 0.49). bf16 matched fp32 (cosine 1.0000).
- The time measurement was also fixed: `dt` is now in seconds for tok/sec and printed in ms.
- The batch is B = 16, T = 1024 as in the video (the earlier version of this report used B = 4, T = 32).
- In this run the cosine LR schedule, the 2D-only weight decay and gradient accumulation were turned off (commented out, `total_batch_size = B * T`), so only the features up to gradient clipping are active.
