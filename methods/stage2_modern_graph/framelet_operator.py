"""Matrix-free author-style Chebyshev framelets with active [low2, high1, high2]."""
from __future__ import annotations

import numpy as np
from scipy import sparse


class FrameletOperator:
    active_bands = ('low2', 'high1', 'high2')
    highpass_scale_weights = (1., 0.25)

    def __init__(self, adjacency, n: int, lambda_max: float):
        adjacency = sparse.csr_matrix(adjacency, dtype=np.float64)
        self.degree = np.asarray(adjacency.sum(axis=1)).ravel()
        if np.any(self.degree <= 0):
            raise ValueError('Framelet requires positive weighted degree')
        inv = 1 / np.sqrt(self.degree)
        self.laplacian = sparse.eye(len(inv), format='csr') - sparse.diags(inv) @ adjacency @ sparse.diags(inv)
        self.n = n
        self.lambda_max = float(lambda_max)
        self.J = np.log(lambda_max / np.pi) / np.log(2.) + 1
        quadrature = np.linspace(0, np.pi, 500)
        argument = np.pi / 2 * (np.cos(quadrature) + 1)
        self.coefficients = [np.array([2 / np.pi * np.trapezoid(np.cos(k * quadrature) * function(argument / 2), quadrature)
                                      for k in range(n)]) for function in (np.cos, np.sin)]

    def _filter(self, x, high: bool, level: int):
        coefficients = self.coefficients[int(high)]
        scale = 2. ** (-self.J + level - 1)
        t0 = x
        t1 = scale / (np.pi / 2) * (self.laplacian @ x) - x
        result = 0.5 * coefficients[0] * t0 + coefficients[1] * t1
        for k in range(2, self.n):
            t2 = 2 * scale / (np.pi / 2) * (self.laplacian @ t1) - 2 * t1 - t0
            result = result + coefficients[k] * t2
            t0, t1 = t1, t2
        return result

    def analysis(self, x):
        low1 = self._filter(x, False, 1)
        return [self._filter(low1, False, 2), self._filter(x, True, 1), self._filter(low1, True, 2)]

    def adjoint(self, bands):
        low_path = self._filter(bands[0], False, 2) + self._filter(bands[2], True, 2)
        return self._filter(low_path, False, 1) + self._filter(bands[1], True, 1)

    def reference_matrices(self):
        if len(self.degree) > 256:
            raise ValueError('dense matrices are limited to P1 small-graph references')
        return self.analysis(np.eye(len(self.degree)))


def operator_checks(operator, rng):
    n = len(operator.degree)
    x = rng.normal(size=(n, 3))
    bands = [rng.normal(size=(n, 3)) for _ in range(3)]
    analysis = operator.analysis(x)
    lhs = sum(np.sum(a * b) for a, b in zip(analysis, bands))
    rhs = np.sum(x * operator.adjoint(bands))
    zero = np.sqrt(operator.degree)
    high = operator.analysis(zero)[1:]
    constant = np.ones(n)
    return {
        'n': operator.n, 'active_bands': list(operator.active_bands), 'J': float(operator.J),
        'lambda_max': operator.lambda_max,
        'adjoint_relative_error': float(abs(lhs - rhs) / max(1., abs(lhs), abs(rhs))),
        'zero_mode_residual': float(np.linalg.norm(operator.laplacian @ zero) / np.linalg.norm(zero)),
        'highpass_zero_mode_responses': [float(np.linalg.norm(value) / np.linalg.norm(zero)) for value in high],
        'reconstruction_relative_error': float(np.linalg.norm(operator.adjoint(analysis) - x) / np.linalg.norm(x)),
        'constant_auxiliary_residual': float(np.linalg.norm(operator.laplacian @ constant) / np.linalg.norm(constant)),
        'constant_auxiliary_highpass': [float(np.linalg.norm(value) / np.linalg.norm(constant))
                                        for value in operator.analysis(constant)[1:]],
    }
