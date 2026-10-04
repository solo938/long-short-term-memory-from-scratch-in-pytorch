# long-short-term-memory-from-scratch-in-pytorch
Hochreiter and Schmidhuber's 1997 paper, rebuilt in PyTorch. Measure with autograd how a plain recurrent network's gradient dies with the time lag, then build the paper's answer: the constant error carousel, the original gated memory cell, and the modern forget-gate cell verified weight for weight against nn.LSTM on the paper's own benchmarks.
