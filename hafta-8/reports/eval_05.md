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
| Layer Normalization  | **2.4724** |      **2.5027** |

## Layer Normalization Results

**Final Loss:**

* **Train Loss:** 2.4724
* **Validation Loss:** 2.5027

## Results Summary

The Layer Normalization experiment achieved slightly lower training and validation losses than the Residual Connections experiment.

| Metric          | Residual Connections | Layer Normalization |
| :-------------- | -------------------: | ------------------: |
| Train Loss      |               2.4774 |              2.4724 |
| Validation Loss |               2.5090 |              2.5027 |

The results show a small reduction in both losses compared with the previous experiment. However, these values alone do not establish the overall effectiveness of Layer Normalization; the implementation and training configuration should also be considered.
