import numpy as np
from scipy.optimize import minimize
from ekf import EKF


def getCalibrationParameters(mag=47.31551):

    # Get the data
    FILE = "./Data_Log_2023_01_14_21_12_13-CHAOS.csv"

    ekf = EKF(file=FILE)
    ekf.loadSamples()

    B_meas = ekf.getMagnSamples()
    B_meas = np.array(B_meas)
    B_meas_NED = np.array([B_meas[:, 1], B_meas[:, 0], -B_meas[:, 2]]).transpose()

    # Set algorithm options
    mag = mag # magnetic field value in Krakow [µT]
    n = B_meas_NED.shape[0] # length of measured vector
    Ts = 0.01

    # Select initial condition for optimization
    A0 = np.eye(3) # initial diagonal matrix
    b0 = np.mean(B_meas_NED, axis=0) # initial offset equal to mean of measurements
    x0 = np.concatenate([np.diag(A0), np.array([0, 0, 0]), b0])
    # print(x0)

    # Fit bias and matrix slope
    def fitSphere(x, B_meas_NED, mag):
        f = 0
        A = np.array([[x[0], x[3], x[5]], [0, x[1], x[4]], [0, 0, x[2]]])
        b = x[6:9]
        iA = np.linalg.inv(A)
        Q = iA.T @ iA

        # Iterate over samples and calculate mean squared error relative to reference sphere
        for i in range(B_meas_NED.shape[0]):
            h = B_meas_NED[i,:]
            if(np.sum(~np.isnan(h)) == 3):
                f = f + (mag**2 - (h-b).T @ Q @ (h-b))**2

        return f

    options = {'gtol': 1e-10, 'maxiter': 300}
    res = minimize(fitSphere, x0, args=(B_meas_NED, mag), options=options)
    x = res.x

    # Retrieve optimized calibration matrix and offset vector
    A = np.array([[x[0], x[3], x[5]], [0, x[1], x[4]], [0, 0, x[2]]])
    b = x[6:]

    return A, b

def calibrate(B_meas, A, b, verbose=False):

    # Decode calibration matrix into gains and angles between axes
    Ae = A.T @ A
    k = np.sqrt(np.diag(Ae))
    K = np.diag(k)
    T = np.linalg.inv(A @ K)
    alpha = [np.degrees(np.arccos(T[i, j])) for i, j in [(0, 1), (1, 2), (0, 2)]]

    if verbose:
        print(f'Calibration matrix A:\t{A}')
        print(f'Calibration offset b:\t{b}')
        print(f'X axis gain:\t\t{k[0]:.2f}')
        print(f'Y axis gain:\t\t{k[1]:.2f}')
        print(f'Z axis gain:\t\t{k[2]:.2f}')
        print(f'Angle between X and Y axis:\t{alpha[0]:.2f}°')
        print(f'Angle between Y and Z axis:\t{alpha[1]:.2f}°')
        print(f'Angle between X and Z axis:\t{alpha[2]:.2f}°')

    # Calibrate measurement
    n = B_meas.shape[0]
    B_cal = np.zeros((n, 3))
    invA = np.linalg.inv(A)
    for i in range(n):
        B_cal[i, :] = invA @ (B_meas[i, :] - b)

    return B_cal