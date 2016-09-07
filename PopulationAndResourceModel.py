
from __future__ import division

import BaseODEs

import PyViability as viab

import numpy as np
import numpy.linalg as la
import warnings as warn

#import matplotlib as mpl
#mpl.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patch

from scipy.optimize import broyden2 as solveZero

import sys, os.path

from myPhaseSpaceL import plotPhaseSpace, savePlot

from SODEoneL import SODEone

from PTopologyL import *

import time

import numba as nb

pi = np.pi

stylePoint["markersize"] *= 2


def easter_rhs(xy, t, phi, r, gamma, delta, kappa):
    x, y = xy
    dx = delta * x + phi * gamma * x * y
    dy = r * y * (1 - y / kappa) - gamma * x * y
    return [dx, dy]



def easter_sunny(xy, xMinimal = None, yMinimal = None):
    # return (xy[:, 0] > xMinimal) & (xy[:, 1] > yMinimal)
    return (xy[:, 0]>xMinimal).__and__(xy[:, 1]>yMinimal)


