# Gradient Accumulation

## 1. Setup

Same as eval_05 (random init `GPT(GPTConfig())`, betas (0.9, 0.95), eps 1e-8, gradient clipping 1.0, cosine LR with warmup, weight decay 0.1 on 2D parameters, fused AdamW, bf16, `torch.compile`, 50 steps) plus gradient accumulation.

| Setting | Value |
|---|---|
| `total_batch_size` | 524,288 tokens (2^19 ≈ 0.5M, the GPT-3 Small batch size) |
| Micro batch | B = 16, T = 1024 → 16,384 tokens per forward/backward |
| `grad_accum_steps` | 524,288 / 16,384 = **32** |
| Optimizer steps | 50 (each one = 32 forward/backward passes) |

Log output:

```
total desired batch size: 524288
=> calculated gradient accumulation steps: 32
```

## 2. Change

GPT-3 Small was trained with a batch of about 0.5M tokens, but 524,288 tokens do not fit on one GPU at once (that would be B = 512 sequences of 1024 tokens). Gradient accumulation simulates this big batch with small ones:

1. `optimizer.zero_grad()` is called once per optimizer step, not once per micro batch.
2. 32 micro batches of 16 × 1024 tokens are processed one after another, and `backward()` is called after each one. The gradients add up in `.grad`.
3. Only after the 32nd micro batch are the gradients clipped, the LR set and `optimizer.step()` called, once.

`tokens_processed` is now `B * T * grad_accum_steps`, and the printed loss is `loss_accum`, the average loss over all 32 micro batches.

## 3. Why Do We Divide the Loss by `grad_accum_steps`?

The loss that `F.cross_entropy` returns is an **average**: the mean loss over the 16 × 1024 = 16,384 tokens of one micro batch. `loss.backward()`, however, does not replace the old gradient. It **adds** the new gradient on top of what is already in `.grad`.

So if we run 32 micro batches without any scaling, `.grad` ends up holding 32 averages added together. That is 32 times the gradient we actually want, which is the gradient of a single average over all 524,288 tokens. Dividing each micro batch loss by 32 before `backward()` turns this sum back into an average:

```
∇(L_1 / 32) + ∇(L_2 / 32) + ... + ∇(L_32 / 32) = ∇((L_1 + L_2 + ... + L_32) / 32) = ∇L_big
```

All micro batches have the same number of tokens, so the mean of the 32 micro batch means is exactly the mean over the 524,288 tokens. The result is the same gradient we would get if the whole batch fit on the GPU at once. In other words, gradient accumulation only saves memory and does not change the math.

If we skipped the division, the gradient would still point in the right direction, but it would be 32 times too long:

- The gradient norm would be 32× larger (about 864 instead of 27 at step 0), so `clip_grad_norm_(..., 1.0)` would clip on almost every step. The gradient would be rescaled by a different factor on every step, and the 1.0 threshold would lose its meaning.
- With plain SGD this would act like a 32× larger learning rate, so `max_lr = 6e-4` would no longer mean what it means for a real 0.5M-token batch. AdamW partly cancels a constant scale because it divides by the running size of the gradients, but the clipping above still breaks the match.
- If we summed the undivided losses for logging, the printed loss would be about 350 at step 0 instead of about 10.9. This is why `loss_accum` is built from the divided losses: the printed value is the average loss of the whole 524,288-token batch.

### Numerical check

To check this, the gradient of one batch was compared with accumulated micro batch gradients on the same tokens (fp32 with TF32 off, K = 4: one batch of 4 × 1024 vs 4 micro batches of 1 × 1024):

| Method | Loss | Gradient norm | Compared with the big batch |
|---|---:|---:|---|
| One batch of 4 × 1024 | 10.928757 | 30.425848 | reference |
| 4 micro batches, `loss / 4` | 10.928757 | 30.425844 | relative error 1.05e-6, cosine similarity 1.00000000 |
| 4 micro batches, no division | 43.715027 (sum) | 121.703377 | same direction (cosine 1.0), but the norm is **4.000000×** larger |

With the division, the accumulated gradient matches the big batch gradient up to floating point rounding (1e-6). Without it, the direction is the same but the gradient is exactly K = 4 times too large.

## 4. Training Log (every step)

| Step | Loss | LR | Grad Norm | dt (ms) | tok/sec |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.9386 | 6.0000e-05 | 27.0126 | 6,170.03 | 84,973 |
| 1 | 9.6493 | 1.2000e-04 | 9.5176 | 3,926.38 | 133,530 |
| 2 | 9.2256 | 1.8000e-04 | 5.7293 | 3,891.83 | 134,715 |
| 3 | 9.8131 | 2.4000e-04 | 8.2065 | 3,896.77 | 134,544 |
| 4 | 9.1916 | 3.0000e-04 | 4.2994 | 3,897.60 | 134,516 |
| 5 | 8.6780 | 3.6000e-04 | 3.6286 | 3,894.64 | 134,618 |
| 6 | 8.2950 | 4.2000e-04 | 1.9536 | 3,900.09 | 134,430 |
| 7 | 8.0680 | 4.8000e-04 | 2.8525 | 3,892.43 | 134,694 |
| 8 | 7.7142 | 5.4000e-04 | 1.9107 | 3,895.59 | 134,585 |
| 9 | 7.3471 | 6.0000e-04 | 1.8004 | 3,896.64 | 134,549 |
| 10 | 7.0297 | 6.0000e-04 | 1.8389 | 3,904.96 | 134,262 |
| 11 | 6.7412 | 5.9917e-04 | 1.5061 | 3,946.41 | 132,852 |
| 12 | 6.5289 | 5.9668e-04 | 1.1516 | 3,916.47 | 133,867 |
| 13 | 6.3773 | 5.9254e-04 | 1.0568 | 3,898.66 | 134,479 |
| 14 | 6.3413 | 5.8679e-04 | 2.6014 | 3,897.06 | 134,534 |
| 15 | 6.2435 | 5.7945e-04 | 0.9687 | 3,897.27 | 134,527 |
| 16 | 6.2126 | 5.7057e-04 | 0.7733 | 3,904.13 | 134,290 |
| 17 | 6.2101 | 5.6021e-04 | 1.1345 | 3,917.23 | 133,841 |
| 18 | 6.1568 | 5.4843e-04 | 1.0178 | 3,938.26 | 133,127 |
| 19 | 6.1473 | 5.3531e-04 | 1.4972 | 3,918.09 | 133,812 |
| 20 | 6.1217 | 5.2092e-04 | 1.0069 | 3,916.57 | 133,864 |
| 21 | 6.0650 | 5.0535e-04 | 0.7380 | 3,940.48 | 133,052 |
| 22 | 6.0527 | 4.8870e-04 | 0.5114 | 3,956.23 | 132,522 |
| 23 | 6.0059 | 4.7107e-04 | 0.5390 | 3,963.13 | 132,291 |
| 24 | 5.9920 | 4.5258e-04 | 0.3715 | 3,960.28 | 132,387 |
| 25 | 5.9803 | 4.3332e-04 | 0.4085 | 3,972.90 | 131,966 |
| 26 | 5.9579 | 4.1343e-04 | 0.3727 | 3,976.78 | 131,837 |
| 27 | 5.9713 | 3.9303e-04 | 0.5005 | 3,982.34 | 131,653 |
| 28 | 5.9408 | 3.7224e-04 | 0.3463 | 3,979.67 | 131,742 |
| 29 | 5.9348 | 3.5118e-04 | 0.3145 | 3,988.65 | 131,445 |
| 30 | 5.9287 | 3.3000e-04 | 0.3090 | 3,989.51 | 131,417 |
| 31 | 5.9061 | 3.0882e-04 | 0.3089 | 3,984.43 | 131,584 |
| 32 | 5.9127 | 2.8776e-04 | 0.3286 | 3,995.09 | 131,233 |
| 33 | 5.8814 | 2.6697e-04 | 0.4399 | 4,005.74 | 130,884 |
| 34 | 5.8732 | 2.4657e-04 | 0.3539 | 4,020.88 | 130,391 |
| 35 | 5.8727 | 2.2668e-04 | 0.4707 | 4,019.26 | 130,444 |
| 36 | 5.8526 | 2.0742e-04 | 0.3436 | 4,022.54 | 130,338 |
| 37 | 5.8690 | 1.8893e-04 | 0.2948 | 4,007.80 | 130,817 |
| 38 | 5.8395 | 1.7130e-04 | 0.2019 | 4,012.32 | 130,670 |
| 39 | 5.8368 | 1.5465e-04 | 0.2406 | 4,033.50 | 129,983 |
| 40 | 5.8377 | 1.3908e-04 | 0.2186 | 4,013.13 | 130,643 |
| 41 | 5.8207 | 1.2469e-04 | 0.2626 | 4,023.36 | 130,311 |
| 42 | 5.8390 | 1.1157e-04 | 0.3199 | 4,018.09 | 130,482 |
| 43 | 5.8120 | 9.9787e-05 | 0.2586 | 4,025.78 | 130,233 |
| 44 | 5.8096 | 8.9428e-05 | 0.2040 | 4,026.51 | 130,209 |
| 45 | 5.8103 | 8.0553e-05 | 0.1963 | 4,016.31 | 130,540 |
| 46 | 5.7954 | 7.3215e-05 | 0.2239 | 4,035.96 | 129,904 |
| 47 | 5.8149 | 6.7460e-05 | 0.1798 | 4,038.59 | 129,820 |
| 48 | 5.7881 | 6.3324e-05 | 0.1761 | 4,034.27 | 129,958 |
| 49 | 5.7875 | 6.0832e-05 | 0.1634 | 4,029.26 | 130,120 |

## 5. Summary

| Metric | eval_05 (no accumulation) | This run (32 micro batches) |
|---|---:|---:|
| `grad_accum_steps` | 1 | 32 |
| Tokens per optimizer step | 16,384 | 524,288 |
| Tokens seen in 50 steps | 819,200 | 26,214,400 |
| Initial loss | 10.9473 | 10.9386 |
| Loss at step 19 | 6.3743 | 6.1473 |
| Final loss (step 49) | 5.8772 | 5.7875 |
| Lowest loss | 5.8052 (step 44) | 5.7875 (step 49) |
| Gradient norm at step 0 | 28.5687 | 27.0126 |
| Mean gradient norm (last 10 steps) | 0.81 | 0.22 |
| Steps clipped (norm > 1.0) | 29 / 50 | 19 / 50 |
| Average step time (steps 1–49) | 125.57 ms | 3,963.67 ms |
| Average throughput (steps 1–49) | ~130,517 tok/sec | ~132,296 tok/sec |
| Total time for 50 steps | 8.4 s | 200.4 s |

## 6. Observations

- The gradient norm at step 0 is 27.01, close to the 28.57 of a single micro batch. This shows the division works: without it the norm would be about 32× larger.
- The loss curve is much smoother. Every optimizer step now averages 32 micro batches, which is 1.6 passes over the whole dataset, so the once-per-epoch bumps of eval_05 are gone. After the warmup bump at step 3 the loss rises by at most 0.02 between steps, while in eval_05 it jumped by up to 0.23 (6.08 → 6.31 at step 25).
- The gradient is much less noisy: the mean norm over the last 10 steps is 0.22 vs 0.81, and only 19 steps were clipped vs 29 (none of the last 20 steps, vs 2).
- With the same 50 steps the loss is lower (5.79 vs 5.88 at step 49, 6.15 vs 6.37 at step 19), but not 32× better. The model still makes only 50 updates with the same LR schedule: a bigger batch gives a more accurate gradient, not more updates.
- Each step takes about 32× longer (3.96 s vs 0.126 s), while throughput stays the same (~132K vs ~130.5K tok/sec). Gradient accumulation does not make the GPU faster; it trades time for memory, since only one 16 × 1024 micro batch has to fit in memory at a time.
- Tiny Shakespeare is small (338,024 tokens): 50 steps of 524,288 tokens are 80 passes over the same text. A 0.5M-token batch only makes real sense with a much larger dataset; here the run mainly shows that the accumulation is implemented correctly.

## 7. Implementation

```python
B = 16 # micro batch size
T = 1024 # sequence length
total_batch_size = 524288 # 2**19 ~0.5M tokens (GPT-3 Small)
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T) # 32

for i in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # sum of 32 gradients -> mean
        loss_accum += loss.detach()     # only for logging
        loss.backward()                 # gradients are added (+=) into .grad
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    lrs.append(lr)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
```
