# Model Evaluation Results

## Loss Comparison

| Experiment           | Train Loss | Validation Loss |
| :------------------- | ---------: | --------------: |
| Task 1               |     2.5602 |          2.5673 |
| Self-Attention       |     2.3809 |          2.4020 |
| Multi-Head Attention |     2.2501 |          2.2753 |
| Feed-Forward         |     2.2151 |          2.2427 |
| Transformer Block    |     2.2093 |          2.2247 |
| Residual Connections |     2.4774 |          2.5090 |

## Residual Connections Results

**Final Loss:**

* **Train Loss:** 2.4774
* **Validation Loss:** 2.5090

## Results Summary

The Residual Connections experiment yielded higher training and validation losses than the previous Transformer Block experiment.

| Metric          | Transformer Block | Residual Connections |
| :-------------- | ----------------: | -------------------: |
| Train Loss      |            2.2093 |               2.4774 |
| Validation Loss |            2.2247 |               2.5090 |

In this experiment, adding residual connections did not reduce the loss. The result alone does not establish that residual connections are ineffective; implementation details, initialization, normalization, and training configuration may affect performance.
