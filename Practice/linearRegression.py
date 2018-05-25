import numpy as np
import matplotlib.pylot as plt

def estimate_coef(x, y):
    # number of observations
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x - n*m_y*m_x)
    SS_xx = np.sum(x*x - n*m_y*m_x)

    # calculating regression coefficients
    b_1 = SS_xy/SS_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)