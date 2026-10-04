"""
Long Short-Term Memory from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rnn_forward
import torch
import torch.nn as nn

def init_rnn(d, h, seed, scale=0.5):

    torch.manual_seed(seed)

    Wx = torch.randn(h, d) * scale
    Wh = torch.randn(h, h) * scale
    b = torch.zeros(h)

    params = {
        'Wx': nn.Parameter(Wx),
        'Wh': nn.Parameter(Wh),
        'b': nn.Parameter(b)
    }

    return params


def rnn_step(x, h_prev, p):

    Wx = p['Wx']
    Wh = p['Wh']
    b = p['b']

    input_term = x @ Wx.T
    hidden_term = h_prev @ Wh.T

    total = input_term + hidden_term + b

    h_next = torch.tanh(total)

    return h_next


def rnn_forward(X, p):

    B, T, d = X.shape
    h = p['Wh'].shape[0]

    h_prev = torch.zeros(B, h, device=X.device, dtype=X.dtype)

    outputs = []

    for t in range(T):

        x_t = X[:, t, :]

        h_next = rnn_step(x_t, h_prev, p)

        h_prev = h_next

        outputs.append(h_next)

    output = torch.stack(outputs, dim=1)

    return output

