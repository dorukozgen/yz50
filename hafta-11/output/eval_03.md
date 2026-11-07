# Gradient Norm Clipping

## Change
Gradient norm clipping was added to the training loop, applied after backpropagation and before the optimizer update to limit excessive gradient magnitudes.

## Training Results (50 Steps)

| Metric | Value |
|---|---:|
| Total steps | 50 |
| Initial loss | 5.1029 |
| Final loss | 5.0346 |
| Minimum gradient norm | 12.5266 |
| Maximum gradient norm | 47.4995 |
| Peak learning rate | 0.0006 |

## Implementation

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=max_norm
)
```