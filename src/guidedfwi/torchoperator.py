__all__ = [
    "TorchOperator",
    "TorchOperatorMulti",
]

import numpy as np
import torch
from functools import partial

class _TorchOperator(torch.autograd.Function):
    """
    Wrapper class for Devito operators into Torch functions.
    """

    @staticmethod
    def forward(ctx, x, propagator, devicetorch):
        ctx.propagator = propagator
        ctx.devicetorch = devicetorch

        # Bring x to CPU and numpy
        x = x.cpu().detach().numpy()

        # Apply forward operator
        loss, grad = ctx.propagator(x)

        # Prepare output
        ctx.grad = torch.from_numpy(grad.reshape(x.shape))
        y = torch.from_numpy(np.array(loss)).to(ctx.devicetorch)

        return y

    @staticmethod
    def backward(ctx, y):
        # Get the pre-computed gradient
        x = ctx.grad.to(ctx.devicetorch)

        return x, None, None, None

class TorchOperator:
    """
    Wrap a Devito operator into a Torch function.
    """

    def __init__(self, prop, devicetorch="cpu", kwargs_prop=None):
        self.prop = prop
        self.devicetorch = devicetorch
        self.kwargs_prop = kwargs_prop
        self.Top = _TorchOperator.apply

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        """Apply forward pass to input vector."""
        return self.Top(x, partial(self.prop, **self.kwargs_prop), self.devicetorch)

class _TorchOperatorMulti(torch.autograd.Function):
    """
    Wrapper class for Devito operators into Torch functions for multiple inputs.
    """

    @staticmethod
    def forward(ctx, *inputs):
        ctx.propagator = inputs[-2]  # The propagator is the second to last input
        ctx.devicetorch = inputs[-1]  # The last input is the device

        # Bring inputs to CPU and numpy
        x = [input.cpu().detach().numpy() for input in inputs[:-2]]

        # Apply forward operator
        loss, grad1, grad2, grad3 = ctx.propagator(*x)  # Unpack inputs for the propagator

        # Prepare output
        ctx.grads = [torch.from_numpy(grad1.reshape(x[0].shape)),
                     torch.from_numpy(grad2.reshape(x[1].shape)),
                     torch.from_numpy(grad3.reshape(x[2].shape))]
        y = torch.from_numpy(np.array(loss)).to(ctx.devicetorch)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Get the pre-computed gradients
        grads = [g.to(ctx.devicetorch) for g in ctx.grads]
        
        # Return gradients for each input
        return tuple(grads) + (None, None) 

class TorchOperatorMulti:
    """
    Wrap a Devito operator into a Torch function for multiple inputs.
    """

    def __init__(self, prop, devicetorch="cpu", kwargs_prop=None):
        self.prop = prop
        self.devicetorch = devicetorch
        self.kwargs_prop = kwargs_prop
        self.Top = _TorchOperatorMulti.apply

    def __call__(self, *inputs):
        return self.apply(*inputs)

    def apply(self, *inputs):
        """Apply forward pass to input vectors."""
        return self.Top(*inputs, partial(self.prop, **self.kwargs_prop), self.devicetorch)