# Weight Decay (only 2D parameters)

## 1. Setup

Same as eval_04 (random init `GPT(GPTConfig())`, B = 16, T = 1024, betas (0.9, 0.95), eps 1e-8, gradient clipping 1.0, cosine LR with warmup, bf16, `torch.compile`, 50 steps). The optimizer is now created by `model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)`. Gradient accumulation is still off.

## 2. Change

The parameters are split into two groups by their number of dimensions:

- **`p.dim() >= 2` → weight decay 0.1:** the weight matrices of every `nn.Linear` (attention `c_attn` and `c_proj`, MLP `c_fc` and `c_proj`) and the embeddings (`wte`, `wpe`). These are the weights used in matrix multiplications.
- **`p.dim() < 2` → weight decay 0.0:** all biases and the LayerNorm weights and biases. They are 1D shift and scale values, so pulling them toward zero gives no useful regularization and can even hurt (for example, a LayerNorm gain shrinking toward 0).

Weight decay is a regularizer. At every step it pulls the weights a little toward zero, so the network cannot rely on a few very large weights and has to spread the work over many of them. AdamW applies the decay separately from the gradient update (`p = p - lr * weight_decay * p`).

The fused AdamW kernel is also turned on when it is available (`fused=True` on CUDA). It updates all parameters in a single kernel instead of launching many small kernels.

## 3. How Many Parameters Get Weight Decay?

Output of `configure_optimizers`:

```
num decayed parameter tensors: 50, with 124,354,560 parameters
num non-decayed parameter tensors: 98, with 121,344 parameters
using fused AdamW: True
```

| Group | Tensors | Parameters | Share of 124,475,904 |
|---|---:|---:|---:|
| Weight decay = 0.1 (2D) | **50** | **124,354,560** | 99.90% |
| Weight decay = 0.0 (1D) | 98 | 121,344 | 0.10% |

Breakdown of the decayed tensors (`wte` and `lm_head` are the same tensor because of weight tying, so it is counted once):

| Parameter | Tensors | Shape | Parameters |
|---|---:|---|---:|
| `transformer.wte.weight` (= `lm_head.weight`) | 1 | (50304, 768) | 38,633,472 |
| `transformer.wpe.weight` | 1 | (1024, 768) | 786,432 |
| `h.*.attn.c_attn.weight` | 12 | (2304, 768) | 21,233,664 |
| `h.*.attn.c_proj.weight` | 12 | (768, 768) | 7,077,888 |
| `h.*.mlp.c_fc.weight` | 12 | (3072, 768) | 28,311,552 |
| `h.*.mlp.c_proj.weight` | 12 | (768, 3072) | 28,311,552 |
| **Total** | **50** | | **124,354,560** |

Breakdown of the non-decayed tensors:

| Parameter | Tensors | Shape | Parameters |
|---|---:|---|---:|
| `h.*.ln_1.weight`, `h.*.ln_1.bias` | 24 | (768,) | 18,432 |
| `h.*.attn.c_attn.bias` | 12 | (2304,) | 27,648 |
| `h.*.attn.c_proj.bias` | 12 | (768,) | 9,216 |
| `h.*.ln_2.weight`, `h.*.ln_2.bias` | 24 | (768,) | 18,432 |
| `h.*.mlp.c_fc.bias` | 12 | (3072,) | 36,864 |
| `h.*.mlp.c_proj.bias` | 12 | (768,) | 9,216 |
| `ln_f.weight`, `ln_f.bias` | 2 | (768,) | 1,536 |
| **Total** | **98** | | **121,344** |

## 4. Training Log (every step)

| Step | Loss | LR | Grad Norm | dt (ms) | tok/sec |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.9473 | 6.0000e-05 | 28.5687 | 2,242.00 | 7,308 |
| 1 | 9.5024 | 1.2000e-04 | 10.6576 | 116.28 | 140,904 |
| 2 | 9.2515 | 1.8000e-04 | 7.3714 | 123.93 | 132,206 |
| 3 | 9.7624 | 2.4000e-04 | 6.6551 | 123.85 | 132,287 |
| 4 | 9.1050 | 3.0000e-04 | 4.2449 | 124.43 | 131,675 |
| 5 | 8.7951 | 3.6000e-04 | 3.2620 | 125.41 | 130,644 |
| 6 | 8.5860 | 4.2000e-04 | 2.3592 | 124.19 | 131,922 |
| 7 | 8.2657 | 4.8000e-04 | 2.0505 | 123.97 | 132,161 |
| 8 | 7.8886 | 5.4000e-04 | 2.2144 | 123.57 | 132,587 |
| 9 | 7.5200 | 6.0000e-04 | 2.1464 | 122.58 | 133,661 |
| 10 | 7.1469 | 6.0000e-04 | 1.8088 | 124.76 | 131,324 |
| 11 | 6.8937 | 5.9917e-04 | 1.2599 | 124.12 | 131,998 |
| 12 | 6.6127 | 5.9668e-04 | 1.0859 | 124.83 | 131,255 |
| 13 | 6.6528 | 5.9254e-04 | 1.1977 | 127.09 | 128,921 |
| 14 | 6.6731 | 5.8679e-04 | 1.1500 | 126.00 | 130,028 |
| 15 | 6.5139 | 5.7945e-04 | 1.4061 | 125.99 | 130,038 |
| 16 | 6.5294 | 5.7057e-04 | 1.1987 | 125.35 | 130,707 |
| 17 | 6.5221 | 5.6021e-04 | 0.8650 | 124.14 | 131,982 |
| 18 | 6.5540 | 5.4843e-04 | 1.2098 | 123.44 | 132,730 |
| 19 | 6.3743 | 5.3531e-04 | 1.6325 | 123.78 | 132,363 |
| 20 | 6.5027 | 5.2092e-04 | 1.5275 | 126.97 | 129,042 |
| 21 | 6.1950 | 5.0535e-04 | 1.2366 | 127.57 | 128,433 |
| 22 | 6.2695 | 4.8870e-04 | 1.1967 | 125.94 | 130,091 |
| 23 | 6.1720 | 4.7107e-04 | 1.2589 | 123.87 | 132,268 |
| 24 | 6.0834 | 4.5258e-04 | 1.1189 | 124.25 | 131,866 |
| 25 | 6.3086 | 4.3332e-04 | 1.0413 | 126.22 | 129,806 |
| 26 | 6.3943 | 4.1343e-04 | 1.3117 | 126.18 | 129,846 |
| 27 | 6.2583 | 3.9303e-04 | 1.2965 | 126.54 | 129,479 |
| 28 | 6.1642 | 3.7224e-04 | 0.7359 | 125.88 | 130,153 |
| 29 | 6.0675 | 3.5118e-04 | 0.8846 | 125.70 | 130,347 |
| 30 | 6.0360 | 3.3000e-04 | 0.7574 | 128.37 | 127,628 |
| 31 | 6.0386 | 3.0882e-04 | 0.9281 | 126.43 | 129,588 |
| 32 | 5.9613 | 2.8776e-04 | 0.7889 | 126.84 | 129,167 |
| 33 | 6.0994 | 2.6697e-04 | 0.7788 | 123.78 | 132,359 |
| 34 | 6.2277 | 2.4657e-04 | 0.7846 | 125.44 | 130,616 |
| 35 | 6.0910 | 2.2668e-04 | 0.8057 | 128.61 | 127,389 |
| 36 | 6.0452 | 2.0742e-04 | 0.9125 | 126.16 | 129,870 |
| 37 | 6.0386 | 1.8893e-04 | 0.7100 | 126.85 | 129,157 |
| 38 | 6.0623 | 1.7130e-04 | 0.9561 | 126.49 | 129,525 |
| 39 | 5.9239 | 1.5465e-04 | 1.0235 | 129.43 | 126,582 |
| 40 | 6.1280 | 1.3908e-04 | 0.7833 | 126.18 | 129,845 |
| 41 | 5.8778 | 1.2469e-04 | 1.0854 | 125.40 | 130,657 |
| 42 | 6.0023 | 1.1157e-04 | 0.8343 | 125.27 | 130,794 |
| 43 | 5.8595 | 9.9787e-05 | 0.8858 | 126.99 | 129,014 |
| 44 | 5.8052 | 8.9428e-05 | 0.8335 | 129.21 | 126,797 |
| 45 | 6.0123 | 8.0553e-05 | 0.8535 | 125.92 | 130,110 |
| 46 | 6.1478 | 7.3215e-05 | 0.7303 | 125.56 | 130,489 |
| 47 | 6.0319 | 6.7460e-05 | 0.7305 | 127.53 | 128,469 |
| 48 | 5.9631 | 6.3324e-05 | 0.6051 | 129.15 | 126,858 |
| 49 | 5.8772 | 6.0832e-05 | 0.7600 | 126.33 | 129,697 |

## 5. Summary

| Metric | eval_04 (AdamW default wd = 0.01 on all parameters) | This run (wd = 0.1 on 2D only, fused AdamW) |
|---|---:|---:|
| Initial loss | 10.9473 | 10.9473 |
| Final loss (step 49) | 5.8820 | 5.8772 |
| Lowest loss | 5.8057 (step 44) | 5.8052 (step 44) |
| Mean gradient norm (last 10 steps) | 0.79 | 0.81 |
| Steps clipped (norm > 1.0) | 30 / 50 | 29 / 50 |
| Average step time (steps 1–49) | 129.56 ms | 125.57 ms |
| Average throughput (steps 1–49) | ~126,491 tok/sec | ~130,517 tok/sec |

## 6. Observations

- Weight decay is applied to **50 tensors with 124,354,560 parameters**, which is 99.90% of the model. Only 98 small 1D tensors (121,344 parameters) are left out.
- The loss curve is almost the same as in eval_04 (final 5.8772 vs 5.8820, lowest 5.8052 vs 5.8057). The difference stays below 0.001 for the first 20 steps and is at most ~0.02 after that. This is expected: in one step the decay shrinks the weights by at most `lr * wd` = 6e-4 × 0.1 = 6e-5 (0.006%), so its regularizing effect only shows up over a long training run, not in 50 steps.
- Fused AdamW made the step about 3% faster: 125.57 ms vs 129.56 ms on average (~130.5K vs ~126.5K tok/sec).

## 7. Implementation

```python
def configure_optimizers(self, weight_decay, learning_rate, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
```
