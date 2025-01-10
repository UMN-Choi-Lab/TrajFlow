import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

def crop_center(image, target_size):
    (w, h) = image.size
    left = (w - target_size) / 2
    top = (h - target_size) / 2
    right = (w + target_size) / 2
    bottom = (h + target_size) / 2
    return image.crop((left, top, right, bottom))

images = []
for i in range(3):
    image_dir = f'va_data/row{i + 1}'
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
    for image_file in image_files:
        images.append(crop_center(Image.open(os.path.join(image_dir, image_file)), 250))

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 21))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(np.array(images[i]))
    ax.axis('off')

cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

observed_traj = np.array([[0, 0], [1, 1], [2, 0]])
unobserved_traj = np.array([[0, 1], [1, 2], [2, 1]])

legend_elements = [
    Line2D([0], [0], color='red', lw=6, label='Observed Trajectory'),
    Line2D([0], [0], color='lightcoral', lw=6, label='Unobserved Trajectory')
]

fig.legend(handles=legend_elements, prop={'size': 12})

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(sm, cax=cbar_ax, label='Likelihood')
cbar.set_label('Likelihood', fontsize=25)
cbar.ax.tick_params(labelsize=20) 

fig.savefig('spatial_density_evolution.pdf', format='pdf')
plt.show()