# Model Evaluation Results

## 1. Loss Results

| Metric          |   Loss |
| :-------------- | -----: |
| Train Loss      | 0.9672 |
| Validation Loss | 1.5899 |

## 2. Model Configuration

| Parameter                                |                            Value |
| :--------------------------------------- | -------------------------------: |
| Batch Size                               |                               64 |
| Maximum Iterations                       |                            5,000 |
| Evaluation Interval                      |                              500 |
| Learning Rate                            |                             3e-4 |
| Device                                   | CUDA if available, otherwise CPU |
| Evaluation Iterations                    |                              200 |
| Block Size                               |                              256 |
| Embedding Dimension (`n_embd`)           |                              384 |
| Number of Transformer Layers (`n_layer`) |                                6 |
| Number of Attention Heads (`n_head`)     |                                6 |
| Dropout                                  |                              0.2 |

## 3. Output

The generated text is available in [`output.txt`](output.txt).

## 4. Results Summary

The model achieved a training loss of **0.9672** and a validation loss of **1.5899**. The difference of **0.6227** suggests a potential generalization gap.

The generated text can be examined in `output.txt` to assess language coherence, grammatical quality, and the model's ability to reproduce dialogue patterns.