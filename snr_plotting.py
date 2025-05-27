from PsfSimulator import PsfSimulator, PsfDataset
from scripts.plotting import PlotController
import PIL
from PIL import Image
import torchvision.transforms as T


sim = PsfSimulator(snr_mean=10, base_noise=100, img_h=15, img_w=15)
image, target = sim.generate(num_spots=1)
image = image*10000
# Tensor to numpy array
image = image.numpy().astype('uint16')
print(image.max())
print(image.mean())

# Plot the numpy array using matplotlib
image = image[0]
import matplotlib.pyplot as plt
plt.imshow(image, cmap='gray')
plt.show()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt



# image is your 2D numpy array
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get the shape
ny, nx = image.shape

# Create X, Y positions for each bar
_x = np.arange(nx)
_y = np.arange(ny)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

# Set minimum z value
z_min = 0
z = np.full_like(x, z_min)

# The bar heights are the pixel values
dz = image.ravel()

# Width and depth of each bar
dx = dy = 1 * np.ones_like(z)

ax.bar3d(x, y, z, dx, dy, dz, shade=True, color='gray')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Pixel Value')
ax.set_zlim(z_min, z_min + dz.max())
plt.show()