import os
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, y, mu, sigma):
    pos = np.dstack((x, y))
    diff = pos - mu
    inv_sigma = np.linalg.inv(sigma)
    normalization = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))
    exponent = -0.5 * np.sum(np.dot(diff, inv_sigma) * diff, axis=2)
    return normalization * np.exp(exponent)

MIN = -2.5
MAX = 2.5

x = np.linspace(MIN, MAX, 1000)
y = np.linspace(MIN, MAX, 1000)
X, Y = np.meshgrid(x, y)
mu = np.array([0, 0])
sigma = np.array([[1, 0], [0, 1]])
Z = gaussian(X, Y, mu, sigma)


plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=(MIN, MAX, MIN, MAX), origin='lower', cmap='viridis', interpolation='bilinear')
plt.axis('off')

file_name = 'gaussian_visual.png'
if os.path.exists(file_name):
        os.remove(file_name)
plt.savefig(file_name, bbox_inches='tight')

plt.show()