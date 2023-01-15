"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        # BEGIN YOUR SOLUTION
        for param in self.params:
            self.u.setdefault(param, 0)
            self.u[param] = self.u[param] * self.momentum + \
                (param.grad.data + self.weight_decay *
                 param.data) * (1 - self.momentum)
            param.data = param.data - self.lr * self.u[param]
        # END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        # BEGIN YOUR SOLUTION
        for param in self.params:
            self.m.setdefault(param, 0)
            self.m[param] = self.m[param] * self.beta1 + \
                (param.grad.data + self.weight_decay *
                 param.data) * (1 - self.beta1)
            self.v.setdefault(param, 0)
            self.v[param] = self.v[param] * self.beta2 + \
                (param.grad.data + self.weight_decay *
                 param.data) ** 2 * (1 - self.beta2)
            m_norm = self.m[param] / (1 - self.beta1 ** (self.t + 1))
            v_norm = self.v[param] / (1 - self.beta2 ** (self.t + 1))
            param.data = param.data - self.lr * \
                m_norm / (v_norm ** 0.5 + self.eps)
        self.t += 1
        # END YOUR SOLUTION
