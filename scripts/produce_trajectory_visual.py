import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.InD import InD

def crop_image(image_path, crop_box, output_path):
    with Image.open(image_path) as img:
        cropped_img = img.crop(crop_box)
        cropped_img.save(output_path)

def plot_observed_and_unobserved_trajectory(background, min_x, max_x, min_y, max_y, observed_traj, unobserved_traj, crop_box):
    plt.figure(figsize=(10, 8))
    plt.axis('off') 
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect='equal')
    plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='#5DA5DA', linewidth=6, label='Observed Trajectory')
    plt.plot(unobserved_traj[:, 0], unobserved_traj[:, 1], color='#E69F00', linewidth=6, label='Unobserved Trajectory')
    file_name = 'ou_visual.png'
    if os.path.exists(file_name):
        os.remove(file_name)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    crop_image('ou_visual.png', crop_box, 'ou_visual.png')  
    plt.show()

def plot_observed_trajectory(background, min_x, max_x, min_y, max_y, observed_traj, crop_box):
    plt.figure(figsize=(10, 8))
    plt.axis('off') 
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect='equal')
    plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='#5DA5DA', linewidth=6, label='Observed Trajectory')
    file_name = 'o_visual.png'
    if os.path.exists(file_name):
        os.remove(file_name)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    crop_image('o_visual.png', crop_box, 'o_visual.png')  
    plt.show()

ind = InD(
    root="data",
    train_ratio=0.75, 
    train_batch_size=64, 
    test_batch_size=1,
    missing_rate=0)

background = plt.imread('data\\paper_background.png')
fudge_factor = 11.5
ortho_px_to_meter = ind.observation_site1.ortho_px_to_meter * fudge_factor
min_x = 0
max_x = background.shape[1] * ortho_px_to_meter
min_y = background.shape[0] * ortho_px_to_meter
max_y = 0

data = list(ind.observation_site1.test_loader)
input, feature, target = data[0]

observed_traj = ind.observation_site1.denormalize(input[0].numpy())
observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)

unobserved_traj = ind.observation_site1.denormalize(target[0].numpy())
unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)

#crop_box = (125, 0, 625, 475)
crop_box = (375, 0, 1875, 1425)
plot_observed_and_unobserved_trajectory(background, min_x, max_x, min_y, max_y, observed_traj, unobserved_traj, crop_box)
plot_observed_trajectory(background, min_x, max_x, min_y, max_y, observed_traj, crop_box)