import numpy as np

class KF:
    def __init__(self, initial_pos: np.array, initial_vel: np.array, accel_variance: np.array) -> None:
        # Convert inputs to np.array
        initial_pos = np.array(initial_pos)
        initial_vel = np.array(initial_vel)
        accel_variance = np.array(accel_variance)

        # Set private variables
        self._x = np.concatenate((initial_pos.reshape((3,1)),initial_vel.reshape((3,1))))
        self._P = np.eye(6)
        self._a_var = np.array([[accel_variance[0], 0, 0, accel_variance[0], 0, 0],
                                [0, accel_variance[1], 0, 0, accel_variance[1], 0],
                                [0, 0, accel_variance[2], 0, 0, accel_variance[2]],
                                [0, 0, 0, accel_variance[0], 0, 0                ],
                                [0, 0, 0, 0, accel_variance[1], 0                ],
                                [0, 0, 0, 0, 0, accel_variance[2]                ],])

    def predict(self, dt: float) -> None:
        # x_kp1 = F * xhat_k
        # P_kp1 = F * P_k * Ft + G * sigma_a^2 * Gt
        F = np.array([[1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt],
                      [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        G = np.concatenate((np.tile(np.array([0.5*dt**2]),(3,1)), np.tile(np.array([dt]),(3,1))))
        x_new= F.dot(self._x) # <- estimating new state based on physics (no control-input model)

        P_new = F.dot(self._P).dot(F.T) + self._a_var.dot(G).dot(G.T) # <- computing the covariance matrix of the state => covariance of state + covariance of acceleration term (the process noise)

        self._x = x_new
        self._P = P_new

    def update(self, meas_value: np.array, meas_var: np.array) -> None:
        # To calculate:           | z <- measurement, H <- measurement variance (of observation noise)
        # y = z - H * x <- Innovation
        # S = H * P * Ht + R <- Innovation cov
        # K = P * Ht * S^-1 <- Kalman gain
        # x = x + K * y <- new state estimate with measurement
        # P = (I - K * H) * P <- new state estimate uncertainty


        # Convert inputs to np.array explicitly
        z = np.array(meas_value).reshape((3,1))
        R = np.array([[meas_var[0], 0, 0],
                      [0, meas_var[1], 0],
                      [0, 0, meas_var[2]]])

        # Define observation model -> map true state space to observed space (i.e. state space to measured space)
        # Only measuring position, so velocity state variables are not mapped
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        x_new = self._x + K.dot(y)
        P_new = (np.eye(6) - K.dot(H)).dot(self._P)

        self._x = x_new
        self._P = P_new





    @property
    def state_cov(self):
        return self._P
    
    @property
    def state(self):
        return self._x

    @property
    def pos(self):
        return self._x[0:3]
    
    @property
    def vel(self):
        return self._x[3:]
    

