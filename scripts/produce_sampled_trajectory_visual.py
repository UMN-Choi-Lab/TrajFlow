import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.InD import InD
from model.TrajFlow import TrajFlow, CausalEnocder, Flow

def plot_traj(ax, background, min_x, max_x, min_y, max_y, input, sample, zoom_factor):
    observed_traj = ind.observation_site1.denormalize(input.cpu().numpy())
    observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)

    unobserved_traj = ind.observation_site1.denormalize(sample.cpu().detach().numpy())
    unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)

    ax.axis('off') 

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    x_range = max_x - min_x
    y_range = max_y - min_y

    square_side = min(x_range, y_range) / zoom_factor

    ax.set_xlim(center_x + square_side / 2, center_x - square_side / 2)
    ax.set_ylim(center_y - square_side / 2, center_y + square_side / 2)
    ax.set_aspect('equal')

    ax.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect='equal')
    ax.plot(observed_traj[:, 0], observed_traj[:, 1], color='red', linewidth=1.5, label='Observed Trajectory')
    ax.plot(unobserved_traj[:, 0], unobserved_traj[:, 1], color='lightcoral', linewidth=1.5, label='Unobserved Trajectory')

def create_combined_plot(background, min_x, max_x, min_y, max_y, input, samples, top_k_sample):
    fig = plt.figure(figsize=(12, 5))

    zoom_factor = 2

    ax1 = plt.subplot(1, 2, 1)
    plot_traj(ax1, background, min_x, max_x, min_y, max_y, input, samples, zoom_factor)
    ax1.text(0.5, -0.1, 'a) Naive Sampling', ha='center', va='top', transform=ax1.transAxes, fontsize=18)

    ax2 = plt.subplot(1, 2, 2)
    plot_traj(ax2, background, min_x, max_x, min_y, max_y, input, top_k_sample, zoom_factor)
    ax2.text(0.5, -0.1, 'b) Top-K Sampling', ha='center', va='top', transform=ax2.transAxes, fontsize=18)

    handles = [
        plt.Line2D([0], [0], color='red', lw=4, label='Observed Trajectory'),
        plt.Line2D([0], [0], color='lightcoral', lw=4, label='Estimated Trajectory')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.78, 1.0), prop={'size': 12})

    plt.subplots_adjust(top=0.85)
    plt.savefig('samples.pdf', bbox_inches='tight', format='pdf')
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
    flow=Flow.CNF,
    marginal=True).to(device)

traj_flow.load_state_dict(torch.load('v_joint.pt'))

background = plt.imread('data/paper_background.png')
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

z_t0, samples, delta_logpz = traj_flow.sample(input, feature, 100, 20)
logpz_t0, logpz_t1 = traj_flow.log_prob(z_t0, delta_logpz)

max_indices = torch.argmax(logpz_t1, dim=0)
top_k_sample = samples[max_indices, torch.arange(samples.size(1))]

create_combined_plot(background, min_x, max_x, min_y, max_y, input[0], samples[0], top_k_sample)