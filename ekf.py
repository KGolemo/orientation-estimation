import numpy as np


def getRotationMatrix(q):
    c00 = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    # c00 = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    c01 = 2 * (q[1] * q[2] - q[0] * q[3])
    c02 = 2 * (q[1] * q[3] + q[0] * q[2])
    
    c10 = 2 * (q[1] * q[2] + q[0] * q[3])
    c11 = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    # c11 = 1 - 2 * (q[1] * q[1] + q[3] * q[3])
    c12 = 2 * (q[2] * q[3] - q[0] * q[1])
    
    c20 = 2 * (q[1] * q[3] - q[0] * q[2])
    c21 = 2 * (q[2] * q[3] + q[0] * q[1])
    c22 = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    # c22 = 1 - 2 * (q[1] * q[1] + q[2] * q[2])

    rotMat = np.array([[c00, c01, c02], 
                       [c10, c11, c12], 
                       [c20, c21, c22]])
    return rotMat


def getEulerAngles(q):
    roll = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
    pitch = np.degrees(np.arcsin(2 * (q[0] * q[2] - q[3] * q[1])))
    yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    return roll, pitch, yaw


class EKF:
    def __init__(self, quaternion0, bias0, acc_ref, mag_ref, p, Q, R):
        
        if bias0 is None:
            self.xHat = quaternion0.transpose()
        else:
            self.xHat = np.concatenate((quaternion0, bias0)).transpose()

        self.yHatBar = np.zeros(3).transpose()
        self.p = p
        self.Q = Q
        self.R = R
        self.K = None
        self.A = None
        self.B = None
        self.H = None
        self.xHatBar = None
        self.xHatPrev = None
        self.pBar = None
        self.measurement = None
        self.err = None
        self.accelReference = acc_ref
        self.magReference = mag_ref
        self.accelReferenceNormalized = acc_ref / (acc_ref[0] ** 2 + acc_ref[1] ** 2 + acc_ref[2] ** 2) ** 0.5 
        self.magReferenceNormalized =  mag_ref / (mag_ref[0] ** 2 + mag_ref[1] ** 2 + mag_ref[2] ** 2) ** 0.5 

    def normalizeQuaternion(self, q):
        qmag = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)**0.5
        return q / qmag

    def getNormalizedAccVector(self, a):
        accel = np.array(a).transpose()
        accelMag = (accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2) ** 0.5
        return accel / accelMag

    def getNormalizedMagVector(self, m):
        mag = np.array(m).transpose()
        magMag = (mag[0] ** 2 + mag[1] ** 2 + mag[2] ** 2) ** 0.5
        return mag / magMag

    def getJacobian(self, reference):
        q0, q1, q2, q3 = self.xHatPrev[0:4]

        e00 = np.matmul(np.array([q0, q3, -q2]), reference.transpose())
        e01 = np.matmul(np.array([q1, q2, q3]), reference.transpose())
        e02 = np.matmul(np.array([-q2, q1, -q0]), reference.transpose())
        e03 = np.matmul(np.array([-q3, q0, q1]), reference.transpose())

        e10 = np.matmul(np.array([-q3, q0, q1]), reference.transpose())
        e11 = np.matmul(np.array([q2, -q1, q0]), reference.transpose())
        e12 = np.matmul(np.array([q1, q2, q3]), reference.transpose())
        e13 = np.matmul(np.array([-q0, -q3, q2]), reference.transpose())

        e20 = np.matmul(np.array([q2, -q1, q0]), reference.transpose())
        e21 = np.matmul(np.array([q3, -q0, -q1]), reference.transpose())
        e22 = np.matmul(np.array([q0, q3, -q2]), reference.transpose())
        e23 = np.matmul(np.array([q1, q2, q3]), reference.transpose())

        jacobianMatrix = 2 * np.array([[e00, e01, e02, e03],
                                        [e10, e11, e12, e13],
                                        [e20, e21, e22, e23]])
        return jacobianMatrix

    def predictAccMag(self):
        # Accel
        hPrime_a = self.getJacobian(self.accelReferenceNormalized)
        accelBar = np.matmul(getRotationMatrix(self.xHatBar).transpose(), self.accelReferenceNormalized)
        # print(accelBar)

        # Mag
        hPrime_m = self.getJacobian(self.magReferenceNormalized)
        magBar = np.matmul(getRotationMatrix(self.xHatBar).transpose(), self.magReferenceNormalized)
        # print(magBar)

        if self.xHat.shape[0] == 7:
            tmp1 = np.concatenate((hPrime_a, np.zeros((3, 3))), axis=1)
            tmp2 = np.concatenate((hPrime_m, np.zeros((3, 3))), axis=1)
            self.H = np.concatenate((tmp1, tmp2), axis=0)
        
        elif self.xHat.shape[0] == 4:
            self.H = np.concatenate((hPrime_a, hPrime_m), axis=0)

        # return np.matmul(self.H, self.xHatBar) 
        return np.concatenate((accelBar, magBar), axis=0)

    def predict(self, w, dt):
        q = self.xHat[0:4]
        Sq = np.array([[-q[1], -q[2], -q[3]],
                       [ q[0], -q[3],  q[2]],
                       [ q[3],  q[0], -q[1]],
                       [-q[2],  q[1],  q[0]]])

        Sw = np.array([[0, -w[0], -w[1], -w[2]],
                       [w[0], 0, w[2], -w[1]],
                       [w[1], -w[2], 0, w[0]],
                       [w[2], w[1], -w[0], 0]])

        if self.xHat.shape[0] == 7:
            tmp1 = np.concatenate((np.identity(4), -dt / 2 * Sq), axis=1)
            tmp2 = np.concatenate((np.zeros((3, 4)), np.identity(3)), axis=1)
            self.F = np.concatenate((tmp1, tmp2), axis=0)
            self.B = np.concatenate((dt / 2 * Sq, np.zeros((3, 3))), axis=0)
            self.xHatBar = np.matmul(self.F, self.xHat) + np.matmul(self.B, np.array(w).transpose())

        elif self.xHat.shape[0] == 4:
            self.F = np.identity((4)) + dt / 2 * Sw
            self.xHatBar = np.matmul(self.F, self.xHat)
        
        self.xHatBar[0:4] = self.normalizeQuaternion(self.xHatBar[0:4])
        self.xHatPrev = self.xHat

        self.pBar = np.matmul(np.matmul(self.F, self.p), self.F.transpose()) + self.Q

    def correct(self, a, m):
        self.yHatBar = self.predictAccMag()

        tmp1 = np.linalg.inv(np.matmul(np.matmul(self.H, self.pBar), self.H.transpose()) + self.R)   # inv(S) S = H * P * H.T + R
        self.K = np.matmul(np.matmul(self.pBar, self.H.transpose()), tmp1)  # K = P * H.T * inv(S)

        magGuass_B = self.getNormalizedMagVector(m)
        accel_B = self.getNormalizedAccVector(a)

        measurement = np.concatenate((accel_B, magGuass_B), axis=0)
        self.measurement = measurement
        self.err = np.linalg.norm(measurement - self.yHatBar) / np.linalg.norm(measurement)

        self.xHat = self.xHatBar + np.matmul(self.K, measurement - self.yHatBar)
        self.xHat[0:4] = self.normalizeQuaternion(self.xHat[0:4])

        if self.xHat.shape[0] == 7:
            self.p = np.matmul(np.identity(7) - np.matmul(self.K, self.H), self.pBar)
        
        elif self.xHat.shape[0] == 4:
            self.p = np.matmul(np.identity(4) - np.matmul(self.K, self.H), self.pBar)