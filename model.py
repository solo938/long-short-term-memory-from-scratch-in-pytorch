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

# Step 2 - gradient_vs_lag
import math
import numpy as np
import torch

def gradient_vs_lag(forward, X, lags):
    # TODO: grad of the last step's states' sum w.r.t. X; norm at step T-1-lag for each lag
    X = X.clone().detach().requires_grad_(True)
    states = forward(X)
    last_state = states[:, -1, :]
    loss = last_state.sum()
    grad = torch.autograd.grad(loss, X)[0]
    T = X.shape[1]
    norms = []
    for lag in lags:
        t = T - 1 -lag
        grad_at_t = grad[:, t, :]
        norm = torch.norm(grad_at_t)
        norms.append(norm.item())
    return np.array(norms)



def decay_rate(lags, norms):
    # TODO: least-squares slope of log(norm) against lag
    lags = np.asarray(lags)
    norms = np.asarray(norms)

    log_norms = np.log(norms)
    slope, intercept = np.polyfit(lags, log_norms, 1)

    return slope

def half_life(rate):
    # TODO: ln(0.5) / rate for negative rate, inf otherwise
    if rate < 0 :
        return math.log(0.5) / rate
    return math.inf

# Step 3 - carousel
import torch

def carousel_forward(U):
    # TODO: cumulative sum of U along time

    B, T, h = U.shape

    c = torch.zeros(B, h, device=U.device, dtype=U.dtype)
    states = []

    for t in range(T):
        u_t = U[:, t, :]
        c = c + u_t
        states.append(c)

    states = torch.stack(states, dim=1)

    return states


def carousel_gain(T, h, lag):
    # TODO: random U (1, T, h) with grad;
    # backprop sum of c_{T-1};
    # return grad at step T-1-lag, batch 0

    U = torch.randn(1, T, h, requires_grad=True)

    states = carousel_forward(U)

    last_state = states[:, -1, :]

    loss = last_state.sum()

    grad = torch.autograd.grad(loss, U)[0]

    t = T - 1 - lag

    gain = grad[0, t, :]

    return gain


def is_constant_error(T, h):
    # TODO: every lag's gain equals ones

    for lag in range(T):
        gain = carousel_gain(T, h, lag)

        ones = torch.ones(h)

        if not torch.equal(gain, ones):
            return False

    return True

