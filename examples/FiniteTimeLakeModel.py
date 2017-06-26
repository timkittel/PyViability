



import numpy as np




# A = np.array([[2, 0],
             # [0, 1]])
A = np.array([[-0.5, 1],
             [-1, -0.5]])
B = np.copy(A)

dim = 2

B[0, 1] = -B[0,1]
B[1, 0] = -B[1,0]

x_1 = -3.5
y_1 = 1

x_2 = 3.5
y_2 = 1


def rhs_default(z, t=0):
    x, y = z
    q = np.array([ (x - x_1) , (y - y_1) ])
    return np.dot(A, q)
def rhs_default_PS(z, t=0):
    x, y = z
    q = np.array([ (x - x_1) , (y - y_1) ])
    return np.tensordot(A, q, axes=[(1,), (0,)])
    # result = np.empty_like(q)
    # for i in dim:
        # for j in dim:
            # result[i] = A[i, j] * q[j]
    # return result
    # # return np.tensordot(A, q, axes=[(1), (0,)])


def rhs_management(z, t=0):
    x, y = z
    q = np.array([ (x - x_2) , (y - y_2) ])
    result = np.dot(B, q)
    if x < 0:
        result[0] = 0
        result[1] = -1
    # result = np.array([0, -1])
        # return result
    # else:
        # q = np.array([ (x - x_2) , (y - y_2) ])
        # result = np.dot(B, q)
    return result
    # return np.dot(B, q)
    # return np.tensordot(B, q, axes=[(1), (0,)])


def rhs_management_PS(z, t=0):
    x, y = z

    result = np.zeros_like(z)
    mask = x < 0

    result[1][mask] = -1

    q = np.array([ (x[~mask] - x_2) , (y[~mask] - y_2) ])
    p = np.tensordot(B, q, axes=[(1), (0,)])

    result[0][~mask] = p[0]
    result[1][~mask] = p[1]
    return result


def sunny(z):
    return z[:, 1] > 0







