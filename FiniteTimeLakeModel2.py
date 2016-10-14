



import numpy as np




def rhs_default(z, t=0):
    x, y = z
    dxdt = (x + 0.5) * (x - 1) * (2 - x)
    dydt = -y
    return dxdt, dydt


# def rhs_default_PS(z, t=0):
    # x, y = z
    # q = np.array([ (x - x_1) , (y - y_1) ])
    # return np.tensordot(A, q, axes=[(1,), (0,)])


def rhs_management(z, t=0):
    x, y = z
    dxdt = y - x
    dydt = y * (y + 2) * (2 - y)
    return dxdt, dydt


# def rhs_management_PS(z, t=0):
    # x, y = z
# 
    # result = np.zeros_like(z)
    # mask = x < 0
# 
    # result[1][mask] = -1
# 
    # q = np.array([ (x[~mask] - x_2) , (y[~mask] - y_2) ])
    # p = np.tensordot(B, q, axes=[(1), (0,)])
# 
    # result[0][~mask] = p[0]
    # result[1][~mask] = p[1]
    # return result


def sunny(z):
    return np.abs(z[:, 0]) > 1







