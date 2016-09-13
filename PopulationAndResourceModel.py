
from __future__ import division

def easter_rhs(xy, t, phi, r, gamma, delta, kappa):
    x, y = xy
    dx = delta * x + phi * gamma * x * y
    dy = r * y * (1 - y / kappa) - gamma * x * y
    return [dx, dy]



def easter_sunny(xy, xMinimal = None, yMinimal = None):
    # return (xy[:, 0] > xMinimal) & (xy[:, 1] > yMinimal)
    return (xy[:, 0]>xMinimal).__and__(xy[:, 1]>yMinimal)


