
from __future__ import division

import functools as ft
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

A_offset = 600  # pre-industrial level corresponds to A=0

beta_default = 0.03  # 1/yr
beta_DG = beta_default / 2
epsilon = 147.  # USD/GJ
phi = 47.e9  # GJ/GtC
tau_A = 50.  # yr
theta_default = beta_default / (950 - A_offset)  # 1/(yr GJ)
theta_SRM = 0.5 * theta_default

A_PB = 945 - A_offset
W_SF = 4e13  # year 2000 GWP

A_mid = 240
W_mid = 7e13
AW_mid = np.array([A_mid, W_mid])


def AW_rhs(AW, t=0, beta=None, theta=None):
    A, W = AW
    Adot = W / (epsilon * phi) - A / tau_A
    Wdot = (beta - theta * A) * W
    return Adot, Wdot


# def AW_rescaled_rhs(aw, t=0, beta=None, theta=None):
    # a, w = aw
    # A = A_mid * a / (1 - a)
    # W = W_mid * w / (1 - w)
    # Adot, Wdot = AW_rhs((A, W), t=t, beta=beta, theta=theta)
    # adot = Adot * A_mid / (A_mid + A)**2
    # wdot = Wdot * W_mid / (W_mid + W)**2
    # return adot, wdot


def AW_rescaled_rhs(aw, t=0, beta=None, theta=None):
    a, w = aw
    A = A_mid * a / (1 - a)
    W = W_mid * w / (1 - w)

    Adot = W / (epsilon * phi) - A / tau_A
    Wdot = (beta - theta * A) * W

    adot = Adot * A_mid / (A_mid + A)**2
    wdot = Wdot * W_mid / (W_mid + W)**2
    return adot, wdot


def AW_sunny(AW):
    # A is not needed to be rescaled and is the only one used here

    return np.logical_and( AW[:, 0] < A_PB, AW[:, 1] > W_SF)
    # return AW[:, 1] > W_SF
    # return (AW[:, 0] < A_PB) & (AW[:, 1] > W_SF)


def AW_rescaled_sunny(aw):
    AW = AW_mid[None, :] * aw / (1 - aw)
    return AW_sunny(AW)





@np.vectorize
def compactification(x, x_mid):
    if x == 0:
        return 0.
    if x == np.infty:
        return 1.
    return x / (x + x_mid)

@np.vectorize
def inv_compactification(y, x_mid):
    if y == 0:
        return 0.
    if np.allclose(y, 1):
        return np.infty
    return x_mid * y / (1 - y)

def transformed_space(transform, inv_transform,
                      start=0, stop=np.infty, num=12,
                      scale=1,
                      num_minors = 50,
                      endpoint=True,
                      ):
    INFTY_SIGN = u"\u221E"
    add_infty = False
    if stop == np.infty and endpoint:
        add_infty = True
        endpoint = False
        num -= 1

    locators_start = transform(start)
    locators_stop = transform(stop)

    major_locators = np.linspace(locators_start,
                           locators_stop,
                           num,
                           endpoint=endpoint)

    major_formatters = inv_transform(major_locators)
    major_formatters = major_formatters / scale

    _minor_formatters = np.linspace(major_formatters[0], major_formatters[-1], num_minors, endpoint=False)[1:] * scale
    minor_locators = transform(_minor_formatters)
    minor_formatters = np.array([np.nan] * len(minor_locators))

    minor_formatters = minor_formatters / scale

    if add_infty:
        # assume locators_stop has the transformed value for infinity already
        major_locators = np.concatenate((major_locators, [locators_stop]))
        major_formatters = np.concatenate(( major_formatters, [ np.infty ]))

    major_string_formatters = np.zeros_like(major_formatters, dtype="|U10")
    mask_nan = np.isnan(major_formatters)
    if add_infty:
        major_string_formatters[-1] = INFTY_SIGN
        mask_nan[-1] = True
    major_string_formatters[~mask_nan] = np.round(major_formatters[~mask_nan], decimals=2).astype(int).astype("|U10")
    return major_string_formatters, major_locators, minor_formatters, minor_locators


def set_ticks():

    ax = plt.gca()

    transf = ft.partial(compactification, x_mid=A_mid)
    inv_transf = ft.partial(inv_compactification, x_mid=A_mid)
    maj_formatters, maj_locators, min_formatters, min_locators = transformed_space(transf, inv_transf)

    ax.xaxis.set_major_locator(ticker.FixedLocator(maj_locators))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(maj_formatters))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(min_locators))

    transf = ft.partial(compactification, x_mid=W_mid)
    inv_transf = ft.partial(inv_compactification, x_mid=W_mid)
    maj_formatters, maj_locators, min_formatters, min_locators = transformed_space(transf, inv_transf, scale=1e12)

    ax.yaxis.set_major_locator(ticker.FixedLocator(maj_locators))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(maj_formatters))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(min_locators))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)




