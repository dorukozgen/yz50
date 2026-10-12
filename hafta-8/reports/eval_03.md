# Model Evaluation Results

## Loss Comparison

| Experiment           | Train Loss | Validation Loss |
| :------------------- | ---------: | --------------: |
| Task 1               |     2.5602 |          2.5673 |
| Self-Attention       |     2.3809 |          2.4020 |
| Multi-Head Attention |     2.2501 |          2.2753 |
| Feed-Forward         |     2.2151 |          2.2427 |
| Transformer Block    | **2.2093** |      **2.2247** |

## Training Configuration

| Parameter           | Value                         |
| :------------------ | :---------------------------- |
| Iterations          | 10,000                        |
| Learning Rate Decay | Multiply learning rate by 0.1 |
| Decay Condition     | `iter >= 5000`                |

## Results Summary

The Transformer Block experiment achieved the lowest training and validation losses among the experiments recorded so far.

* **Train Loss:** 2.2093
* **Validation Loss:** 2.2247

### Learning Rate Schedule

The model was trained for 10,000 iterations. When the iteration count reached 5,000, the learning rate was multiplied by 0.1.

```python
if iter >= 5000:
    learning_rate *= 0.1
```
