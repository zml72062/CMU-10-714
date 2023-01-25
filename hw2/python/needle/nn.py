"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(
            out_features, 1, device=device, dtype=dtype).reshape((1, out_features))) if bias else None
        # END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        bias_shape = list(X.shape)
        bias_shape[-1] = self.weight.shape[-1]
        return ops.matmul(X, self.weight) + ops.broadcast_to(
            self.bias, bias_shape
        )
        # END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # BEGIN YOUR SOLUTION
        flattened = int(np.array(X.shape).prod()/X.shape[0])
        return ops.reshape(X, (X.shape[0], flattened))
        # END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return ops.relu(x)
        # END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        # END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[1], y)
        return ops.summation((ops.logsumexp(logits, axes=(1, ))
                             - ops.summation(logits * y_one_hot, axes=(1, )))
                             / logits.shape[0],
                             axes=(0, ))
        # END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(
            dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(
            dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(
            dim, device=device, dtype=dtype, requires_grad=False)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        num_sample = x.shape[0]
        if self.training:
            mean = ops.summation(x, axes=(0, )) / num_sample
            self.running_mean.data = (1-self.momentum) * \
                self.running_mean.data + self.momentum * mean.data
            x_unbiased = x - mean.broadcast_to(x.shape)
            var = ops.summation(x_unbiased**2, axes=(0, )) / num_sample
            self.running_var.data = (1-self.momentum) * \
                self.running_var.data + \
                self.momentum * var.data
            return (self.weight / (var + self.eps) ** 0.5).broadcast_to(
                x.shape) * x_unbiased + self.bias.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * (x - self.running_mean.broadcast_to(
            x.shape)) / (self.running_var + self.eps).broadcast_to(x.shape) ** 0.5 + \
            self.bias.broadcast_to(x.shape)
        # END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(
            dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True))
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        num_sample = x.shape[0]
        mean = (ops.summation(x, axes=(1, )) /
                self.dim).reshape((num_sample, 1))
        x_unbiased = x - mean.broadcast_to(x.shape)
        var = ((ops.summation(x_unbiased**2, axes=(1, )) / self.dim
                + self.eps)**0.5).reshape((num_sample, 1))
        return self.weight.broadcast_to(x.shape) / var.broadcast_to(
            x.shape) * x_unbiased + self.bias.broadcast_to(x.shape)
        # END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return x * init.randb(*x.shape, p=1-self.p, dtype='float32') / (1-self.p) \
            if self.training else x
        # END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return self.fn(x) + x
        # END YOUR SOLUTION
