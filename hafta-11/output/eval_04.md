# Cosine Learning Rate Scheduler with Linear Warmup

## Change
Implemented a learning rate scheduler combining linear warmup and cosine decay.

- **Linear Warmup:** Gradually increases the learning rate during initial training steps.
- **Cosine Decay:** Gradually decreases the learning rate after warmup.

## Training Results (50 Steps)

| Metric | Value |
|---|---:|
| Total steps | 50 |
| Initial learning rate | 0.0001 |
| Peak learning rate | 0.0006 |
| Final learning rate | 0.0001 |
| Initial loss | 5.1029 |
| Final loss | 5.0066 |
| Minimum gradient norm | 12.2949 |
| Maximum gradient norm | 60.0960 |
| Average gradient norm | 39.27 |

## Learning Rate Schedule

- Steps 0–9: Linear warmup.
- Steps 10–49: Cosine decay.

The scheduler provides gradual learning rate adjustments throughout training.