# Model Evaluation Results

## Loss Comparison

| Experiment           | Train Loss | Validation Loss |
| :------------------- | ---------: | --------------: |
| Task 1               |     2.5602 |          2.5673 |
| Self-Attention       |     2.3809 |          2.4020 |
| Multi-Head Attention |     2.2501 |          2.2753 |
| Feed-Forward         | **2.2151** |      **2.2427** |

## Results Summary

The Feed-Forward layer further reduced both training and validation loss.

* **Task 1:** Train Loss = 2.5602, Validation Loss = 2.5673
* **Self-Attention:** Train Loss = 2.3809, Validation Loss = 2.4020
* **Multi-Head Attention:** Train Loss = 2.2501, Validation Loss = 2.2753
* **Feed-Forward:** Train Loss = 2.2151, Validation Loss = 2.2427

### Best Result

The Feed-Forward experiment achieved the lowest losses so far:

* **Train Loss:** 2.2151
* **Validation Loss:** 2.2427
