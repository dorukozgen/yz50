# Weight Decay

## Change
Implemented weight decay in the optimizer to discourage excessively large model weights and improve generalization by reducing overfitting.

## Training Results (50 Steps)

| Metric | Value |
|---|---:|
| Total steps | 50 |
| Initial loss | 5.1029 |
| Final loss | 4.9911 |
| Minimum gradient norm | 12.3652 |
| Maximum gradient norm | 56.7286 |
| Average gradient norm | 38.20 |
| Peak learning rate | 0.0006 |
| Final learning rate | 0.0001 |

## Implementation

Weight decay was applied through the optimizer's `weight_decay` parameter.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
```