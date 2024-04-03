from kf import KF
import unittest
import numpy as np
from numpy import testing

class TestKF(unittest.TestCase):
    def test_can_construct(self):

        init_pos = np.array([0,1,2]).reshape((3,1))
        init_vel = np.array([0,1,2]).reshape((3,1))
        accel_var = np.array([0.1,0,0])

        kf = KF(initial_pos=init_pos, initial_vel=init_vel, accel_variance=accel_var)
        print(kf.vel)
        testing.assert_allclose(kf.pos,init_pos)
        testing.assert_allclose(kf.vel, init_vel)
        # self.assertAlmostEqual(kf.pos, init_pos)
        # self.assertAlmostEqual(kf.vel, init_vel)

    def test_predict_state_and_state_cov_are_right_shape(self):
        init_pos = np.array([0,1,2])
        init_vel = np.array([0,1,2])
        accel_var = np.array([0.1,0,0])
        dt = 0.5

        kf = KF(initial_pos=init_pos, initial_vel=init_vel, accel_variance=accel_var)
        kf.predict(dt=dt)

        self.assertEqual(kf.state.shape, (6,1))
        self.assertEqual(kf.state_cov.shape, (6,6))

    def test_predict_increases_state_uncertainty(self):
        # Predictions without measurement updates will increase the uncertaintly of the state
        init_pos = np.array([0,1,2])
        init_vel = np.array([0,1,2])
        accel_var = np.array([0.1,0,0])
        dt = 0.5

        kf = KF(initial_pos=init_pos, initial_vel=init_vel, accel_variance=accel_var)

        for i in range(10):
            det_before = np.linalg.det(kf.state_cov)
            kf.predict(dt=dt)
            det_after = np.linalg.det(kf.state_cov)
            self.assertGreater(det_after, det_before)

    def test_update(self):
        init_pos = np.array([0,1,2])
        init_vel = np.array([0,1,2])
        accel_var = np.array([0.1,0,0])
        dt = 0.5

        kf = KF(initial_pos=init_pos, initial_vel=init_vel, accel_variance=accel_var)

        det_before = np.linalg.det(kf.state_cov)
        kf.update(meas_value=np.array([0,2,3]), meas_var=np.array([0.1, 0., 0.3]))
        det_after = np.linalg.det(kf.state_cov)

        self.assertLess(det_after,det_before)

        
        
