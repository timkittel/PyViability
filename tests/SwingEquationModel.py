

from __future__ import division

import numpy as np


pi = np.pi


def swing_rhs(theta_omega, t=0, P = None, K = None, alpha = None):  # raises an error a not set
    theta, omega = theta_omega
    dtheta = omega
    domega = 2*P - alpha * omega - 2 * K * np.sin(theta)
    return [dtheta, domega]


def swing_sunny(p):
    """sunny constraint for gravity Pendulum"""
    return np.abs(p[:, 1]) < 0.5

