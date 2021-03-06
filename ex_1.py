import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
# 1000 samples from an uni-variant gaussian distribution with mean 1 and a standard deviation of 0.2
mu = 1
sigma = 0.2
n_bins = 30
# experimental distribution
samples = np.sort(np.random.normal(mu, sigma, size=1000))
fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
# density=1 controls data normalization
# n: value of bins
# n_bins: number of bins
# bins: The edges of the bins, edges = n_bins + 1
n, bins, patches = ax1.hist(samples, n_bins, density=1)
# Ground truth distribution
x = np.arange(mu-3*sigma,mu+3*sigma,0.001)
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((samples - mu) / sigma)**2))
ax2 = fig1.add_subplot(122)
ax2.plot(x, norm.pdf(x, mu, sigma), label='ground truth distribution')

# 2D
mean_vec = np.array([0.5, -0.2])
cov = np.array([[2.0, 0.3], [0.3, 0.5]])
n_bins = 30

# theoretical 2d gaussian
x, y = np.mgrid[-6.0:6.0:100j, -3.0:3.0:100j]
# plot 3d figure pos.dim == 3
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
rv = multivariate_normal(mean_vec, cov)
# z axis is the joint pdf of random variables
z = rv.pdf(pos)
fig2 = plt.figure(figsize=(9, 5))
ax3 = fig2.add_subplot(121,projection='3d')
ax3.plot_wireframe(x, y, z)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('PDF')
ax3.set_title('Ground Truth Distribution 2D')

ax4 = fig2.add_subplot(122,projection='3d')
z = np.random.multivariate_normal(mean_vec, cov, 10000)
hist, xedges, yedges = np.histogram2d(z[:, 0], z[:, 1], bins=30, range=[[-6, 6], [-3, 3]], density=True)
# :-1 except the last one, since it is not included
x_pos, y_pos = np.meshgrid(xedges[:-1], yedges[:-1])
# x_pos = x_pos.flatten('F')
# y_pos = y_pos.flatten('F')
# ndarray.Ravel Return a flattened array.
x_pos = x_pos.ravel()
y_pos = y_pos.ravel()
z_pos = np.zeros_like(x_pos)

# Construct arrays with the dimensions for the 100 bars.
# Note that the width of bars should be identical to the distance between two subsequential samples
# dx = np.ones_like(z_pos)
# dy = dx.copy()
dx =  xedges[1] - xedges[0]
dy =  yedges[1] - yedges[0]
dz = hist.flatten()

ax4.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='g', zsort='average')
plt.show()

