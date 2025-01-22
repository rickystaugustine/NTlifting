import numpy as np


def m_func(inputs, a, b, c, d):
    """
    Multiplier function model.
    """
    w, s, r = inputs
    return a * w + b * s + c * np.log(r + 1) + d


class ConstantMultiplier:
    """
    Represents a constant multiplier.
    """
    def __init__(self, value: float):
        self.value = value

    def __call__(self, w: float, s: float, r: float) -> float:
        return self.value


class FittedMultiplier:
    """
    Represents a fitted multiplier function.
    """
    def __init__(self, params: tuple):
        self.params = params

    def __call__(self, w: float, s: float, r: float) -> float:
        return m_func((w, s, r), *self.params)
