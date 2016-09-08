

from __future__ import division, print_function, generators

import numpy as np


pi = np.pi


def techChange_rhs(uB_pB, t, rvar, pBmin, pE, delta, smax, sBmax):
    uB, pB = uB_pB
    if sBmax == 0.:
        p = pE
    else:
        if smax < sBmax * uB:
            p = pE + smax / uB
        else:
            p = sBmax + pE

    duB = rvar * uB * (1 - uB) * (p - pB)
    dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)

    return np.array([duB, dpB])


def techChange_sunny(p):
    """sunny constraint for techChangeModel"""
    return p[:, 0] > 0.325


def techChange_rhsPS(uB_pB, t, rvar, pBmin, pE, delta, smax, sBmax):
    uB, pB = uB_pB
    p = np.zeros_like(pB)
    p[:] = sBmax + pE
    mask = (smax < sBmax * uB)
    p[mask] = (pE + smax / uB[mask])

    duB = rvar * uB * (1 - uB) * (p - pB)
    dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)
    return np.array([duB, dpB])

