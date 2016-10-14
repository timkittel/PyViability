
import numpy as np


def rhs_default(z, t=0):
    x, y = z
    dxdt = (x + 0.5) * (x - 1) * (2 - x)
    dydt = -y
    return dxdt, dydt


def rhs_management(z, t=0):
    x, y = z
    dxdt = y - x
    dydt = y * (y + 2) * (2 - y)
    return dxdt, dydt


def sunny(z):
    return np.abs(z[:, 0]) > 1







