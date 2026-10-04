# Long Short-Term Memory from Scratch in PyTorch

Hochreiter and Schmidhuber's 1997 paper, rebuilt in PyTorch. Measure with autograd how a plain recurrent network's gradient dies with the time lag, then build the paper's answer: the constant error carousel, the original gated memory cell, and the modern forget-gate cell verified weight for weight against nn.LSTM on the paper's own benchmarks.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** rnn_forward
- [x] **2.** gradient_vs_lag
- [x] **3.** carousel

---

Built on Deep-ML.
