import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.InD import InD
from model.TrajFlow import TrajFlow, CausalEnocder, Flow

def crop_center(image, target_size):
    (w, h) = image.size
    left = (w - target_size) / 2
    top = (h - target_size) / 2
    right = (w + target_size) / 2
    bottom = (h + target_size) / 2
    return image.crop((left, top, right, bottom))

def crop_image(image_path, target_size, output_path):
    with Image.open(image_path) as img:
        cropped_img = crop_center(img, target_size)
        cropped_img.save(output_path)

def plot_traj(background, min_x, max_x, min_y, max_y, target_size, input, sample, file_name):
    observed_traj = ind.observation_site1.denormalize(input.cpu().numpy())
    observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)

    unobserved_traj = ind.observation_site1.denormalize(sample.cpu().detach().numpy())
    unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)

    plt.figure(figsize=(10, 8))
    plt.axis('off') 
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect='equal')
    plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='red', linewidth=1, label='Observed Trajectory')
    plt.plot(unobserved_traj[:, 0], unobserved_traj[:, 1], color='lightcoral', linewidth=1, label='Unobserved Trajectory')
    if os.path.exists(file_name):
        os.remove(file_name)
    plt.savefig(file_name, bbox_inches='tight')
    crop_image(file_name, target_size, file_name)  
    plt.show()

ind = InD(
    root="data",
    train_ratio=0.75, 
    train_batch_size=64, 
    test_batch_size=1,
    missing_rate=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

traj_flow = TrajFlow(
		seq_len=100, 
		input_dim=2, 
		feature_dim=5, 
		embedding_dim=128,
		hidden_dim=512,
		causal_encoder=CausalEnocder.CDE,
		flow=Flow.CNF).to(device)
traj_flow.load_state_dict(torch.load('ind_marginal.pt'))

background = plt.imread('data\\paper_background.png')
fudge_factor = 11.5
ortho_px_to_meter = ind.observation_site1.ortho_px_to_meter * fudge_factor
min_x = 0
max_x = background.shape[1] * ortho_px_to_meter
min_y = background.shape[0] * ortho_px_to_meter
max_y = 0

data = list(ind.observation_site1.test_loader)
input, feature, target = data[0]
input = input.to(device)
feature = feature.to(device)

z_t0, samples, delta_logpz = traj_flow.sample(input, feature, 20)
logpz_t0, logpz_t1 = traj_flow.log_prob(z_t0, delta_logpz)

max_indices = torch.argmax(logpz_t1, dim=0)
top_k_sample = samples[max_indices, torch.arange(samples.size(1))]

target_size = 250
plot_traj(background, min_x, max_x, min_y, max_y, target_size, input[0], samples[0], 'naive_sample.png')
plot_traj(background, min_x, max_x, min_y, max_y, target_size, input[0], top_k_sample, 'top_k_sample.png')

naive_sampling_img = Image.open('naive_sample.png')
top_k_sampling_img = Image.open('top_k_sample.png')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(naive_sampling_img)
axes[0].axis('off')

axes[1].imshow(top_k_sampling_img)
axes[1].axis('off')

axes[0].text(0.5, -0.1, 'a) Naive sampling', ha='center', va='top', transform=axes[0].transAxes, fontsize=14)
axes[1].text(0.5, -0.1, 'b) Top-k sampling', ha='center', va='top', transform=axes[1].transAxes, fontsize=14)

handles = [plt.Line2D([0], [0], color='red', lw=4, label='Observed Trajectory'),
           plt.Line2D([0], [0], color='lightcoral', lw=4, label='Estimated Trajectory')]
fig.legend(handles=handles)

plt.savefig('samples.png')
plt.show()