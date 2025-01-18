import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.EthUcy import EthUcy
from model.TrajFlow import TrajFlow, CausalEnocder, Flow

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_trajectories=1

eth = EthUcy(train_batch_size=128, test_batch_size=num_trajectories, history=8, futures=12, smin=0.3, smax=1.7)
observation_site = eth.zara1_observation_site

traj_flow = TrajFlow(
    seq_len=12, input_dim=2, feature_dim=4,
    embedding_dim=128, hidden_dim=512,
    causal_encoder=CausalEnocder.GRU,
    flow=Flow.DNF,
    marginal=True,
    norm_rotation=True).to(device)
traj_flow.load_state_dict(torch.load('trajflow_marginal.pt'))
traj_flow.eval()

data = list(observation_site.test_loader)
input, feature, target = data[0]
input = input.to(device)
feature = feature.to(device)
target = target.to(device)

plt.figure(figsize=(10, 8))
plt.axis('off')

x_center = 0
y_center = 0

for i in range(num_trajectories):
    observed_traj = input[i].cpu().numpy()
    observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)
    #unobserved_traj = target[i].cpu().numpy()
    #unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)
    x_center = observed_traj[-1, 0]
    y_center = -observed_traj[-1, 1]
    plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='#5DA5DA', linewidth=5, label='Observed Trajectory')
    #plt.plot(unobserved_traj[:, 0], unobserved_traj[:, 1], color='#E69F00', linewidth=5, label='Unobserved Trajectory')
    #plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='#E69F00', linewidth=5, label='Observed Trajectory')
    #plt.plot(unobserved_traj[:, 0], unobserved_traj[:, 1], color='#CC79A7', linewidth=5, label='Unobserved Trajectory')
    #plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='black', linewidth=5, label='Observed Trajectory')

steps = 100#1000
batch_size = 1000
grid_range = 5
linspace_x = torch.linspace(x_center - grid_range, x_center + grid_range, steps)
linspace_y = torch.linspace(y_center - grid_range, y_center + grid_range, steps)
x, y = torch.meshgrid(linspace_x, linspace_y)
grid = torch.stack((x.flatten(), y.flatten()), dim=-1).to(device)

likelihoods = []

for i in range(num_trajectories):
    with torch.no_grad():
        x = input[i].unsqueeze(0)
        x_t = x[:, -1:, :]
        y = target[i].unsqueeze(0)
        feat = feature[i].unsqueeze(0)

        embedding = traj_flow._embedding(x, feat)
        embedding = embedding.repeat(batch_size, 1)

        pz_t1 = []
        pool = 0
        for grid_batch in grid.split(batch_size, dim=0):
            pool += 1
            print(pool)
            #grid_batch = grid_batch.unsqueeze(1).expand(-1, 12, -1)
            grid_batch = grid_batch.unsqueeze(1).expand(-1, 120, -1)
            z_t0, delta_logpz = traj_flow.flow(grid_batch, embedding, sampling_frequency=10)
            logpz_t0, logpz_t1 = traj_flow.log_prob(z_t0, delta_logpz)
            pz_t1.append(logpz_t1.exp())
        
        pz_t1 = torch.cat(pz_t1, dim=0)
        fused_probs = torch.sum(pz_t1, dim=-1)
        fused_normalized_probs = fused_probs / torch.max(fused_probs)
        likelihoods.append(fused_normalized_probs)
t = likelihoods[0]

grid_numpy = grid.cpu().detach().numpy()
xx = grid_numpy[:, 0].reshape(steps, steps)
yy = -grid_numpy[:, 1].reshape(steps, steps)
likelihood = t.cpu().numpy().reshape(steps, steps)
#likelihood = np.where(likelihood < 0.01, np.nan, likelihood)
heat_map = plt.pcolormesh(xx, yy, likelihood, shading='auto', cmap=plt.cm.viridis)

cbar = plt.colorbar(heat_map, label='Likelihood', pad=0.05, aspect=20)
cbar.ax.set_ylabel('Likelihood', rotation=270, labelpad=15)
plt.legend(loc='upper left', bbox_to_anchor=(0.92, 1.1), borderaxespad=0)

plt.savefig('occupancy_map.png', dpi=300, bbox_inches='tight')
plt.show()
