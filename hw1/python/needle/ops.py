"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return a ** self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a, = node.inputs
        return (self.scalar * power_scalar(a, self.scalar - 1) * out_grad, )
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a / b
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs ** 2
        # END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a / self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        # END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        # calculate the `axes` argument of array_api.transpose()
        axes = list(range(a.ndim))
        if self.axes is None:
            id1, id2 = a.ndim-1, a.ndim-2
        else:
            id1, id2 = self.axes[0], self.axes[1]
        axes[id1], axes[id2] = id2, id1
        return array_api.transpose(a, axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return (transpose(out_grad, axes=self.axes), )
        # END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a, = node.inputs
        return (reshape(out_grad, a.shape), )
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION

        # find shape of input & output
        in_grad = out_grad
        a, = node.inputs
        old_shape, new_shape = a.shape, self.shape
        old_dim, new_dim = len(old_shape), len(new_shape)
        extra_dim = new_dim - old_dim

        # gradient to input is summation over out_grad on broadcast dimensions

        # if new_shape has more dimensions than old_shape in the front,
        # sum on those dimensions
        for _ in range(extra_dim):
            in_grad = summation(in_grad, axes=0)
        # if old_shape has length 1 on an axis, where new_shape has length
        # more than 1, sum on the axis with keepdims=True
        cumulative_dim = 0
        for i in range(old_dim):
            if old_shape[i] != new_shape[i + extra_dim]:
                in_grad = summation(in_grad, axes=cumulative_dim)
            else:
                cumulative_dim += 1
        return (in_grad.reshape(old_shape), )  # reshape() makes keepdims=True
        # END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a.sum(axis=self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a, = node.inputs
        in_shape = list(a.shape)
        axes = self.axes
        if axes is None:
            axes = range(len(in_shape))
        for i in axes:
            in_shape[i] = 1
        # add dimensions on summed axes, and broadcast to in_shape
        return (broadcast_to(out_grad.reshape(in_shape), a.shape), )
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, transpose(
            rhs)), matmul(transpose(lhs), out_grad)
        # find properly broadcast shape of result
        out_shape, lshape, rshape = out_grad.shape, lhs.shape, rhs.shape
        out_dim, ldim, rdim = len(out_shape), len(lshape), len(rshape)

        # execute backward computation on all broadcast dimensions
        # except for the last two dimensions

        for _ in range(out_dim-ldim):
            lgrad = summation(lgrad, axes=0)
        cumulative_dim = 0
        for i in range(ldim-2):
            if out_shape[i+out_dim-ldim] != lshape[i]:
                lgrad = summation(lgrad, axes=cumulative_dim)
            else:
                cumulative_dim += 1

        for _ in range(out_dim-rdim):
            rgrad = summation(rgrad, axes=0)
        cumulative_dim = 0
        for i in range(rdim-2):
            if out_shape[i+out_dim-rdim] != rshape[i]:
                rgrad = summation(rgrad, axes=cumulative_dim)
            else:
                cumulative_dim += 1

        return lgrad.reshape(lshape), rgrad.reshape(rshape)
        # END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return -a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return (negate(out_grad), )
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.log(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a, = node.inputs
        return (out_grad / a, )
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a, = node.inputs
        return (out_grad * exp(a), )
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        b = array_api.copy(a)
        b[a < 0] = 0
        return b
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a, = node.inputs
        raw = array_api.copy(out_grad.realize_cached_data())
        raw[a.numpy() < 0] = 0
        return (Tensor(raw, device=a.device, dtype=a.dtype), )
        # END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
