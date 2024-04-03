import numpy as np
import matplotlib.pyplot as plt

from kf import KF

plt.ion()
plt.figure()

kf = KF(initial_pos=np.array([0.3, 0.5, 0.2]), initial_vel=np.array([0.4, -0.2, 1]), accel_variance=np.array([1,1,0.1])**2)

DT = 0.1
NUM_STEPS = 1000

MEAS_EVERY_STEPS = 10

real_pos = np.array([0,0,0])
real_vel = np.array([0.5,0.1,0.2])


mus = []
covs = []
real_poses = []
actual_meas_var = np.array([1, 10, 1])**2 # actual variance of measurement noise
meas_var = actual_meas_var #np.array([10, 10, 10])**2 # estimated variance of measurement noise

for step in range(NUM_STEPS):
    covs.append(kf.state_cov)
    mus.append(kf.state)

    real_pos = real_pos + real_vel*DT
    real_poses.append(real_pos)

    kf.predict(dt=DT)

    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value=real_pos+np.random.randn()*np.sqrt(actual_meas_var), meas_var=meas_var )

plt.subplot(2,1,1)
plt.title('Position')
plt.plot([pos[0] for pos in real_poses],'k--')
plt.plot([pos[1] for pos in real_poses],'k--')
plt.plot([pos[2] for pos in real_poses],'k--')
plt.plot([mu[0] for mu in mus],'r')
plt.plot([mu[0]-2*np.sqrt(cov[0,0]) for mu,cov in zip(mus,covs)],'r--')
plt.plot([mu[0]+2*np.sqrt(cov[0,0]) for mu,cov in zip(mus,covs)],'r--')
plt.plot([mu[1] for mu in mus],'b')
plt.plot([mu[1]-2*np.sqrt(cov[1,1]) for mu,cov in zip(mus,covs)],'b--')
plt.plot([mu[1]+2*np.sqrt(cov[1,1]) for mu,cov in zip(mus,covs)],'b--')
plt.plot([mu[2] for mu in mus],'g')
plt.plot([mu[2]-2*np.sqrt(cov[2,2]) for mu,cov in zip(mus,covs)],'g--')
plt.plot([mu[2]+2*np.sqrt(cov[2,2]) for mu,cov in zip(mus,covs)],'g--')

plt.subplot(2,1,2)
plt.title('Velocity')
plt.plot([mu[3] for mu in mus],'r')
plt.plot([mu[3]-2*np.sqrt(cov[3,3]) for mu,cov in zip(mus,covs)],'r--')
plt.plot([mu[3]+2*np.sqrt(cov[3,3]) for mu,cov in zip(mus,covs)],'r--')
plt.plot([mu[4] for mu in mus],'b')
plt.plot([mu[4]-2*np.sqrt(cov[4,4]) for mu,cov in zip(mus,covs)],'b--')
plt.plot([mu[4]+2*np.sqrt(cov[4,4]) for mu,cov in zip(mus,covs)],'b--')
plt.plot([mu[5] for mu in mus],'g')
plt.plot([mu[5]-2*np.sqrt(cov[5,5]) for mu,cov in zip(mus,covs)],'g--')
plt.plot([mu[5]+2*np.sqrt(cov[5,5]) for mu,cov in zip(mus,covs)],'g--')

plt.show()
# plt.ginput(-1)
