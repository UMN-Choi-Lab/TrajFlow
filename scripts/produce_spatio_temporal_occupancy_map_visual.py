import os
import sys
import torch
import numpy as np
import matplotlib
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
    embedding_dim=32, hidden_dim=64,
    causal_encoder=CausalEnocder.CDE,
    flow=Flow.CNF,
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

for i in range(num_trajectories):
    observed_traj = input[i].cpu().numpy()
    observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)
    plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='black', linewidth=5, label='Observed Trajectory')

steps = 400
batch_size = 4000
linspace = torch.linspace(0, 1, steps)
x, y = torch.meshgrid(linspace, linspace)
grid = torch.stack((x.flatten(), y.flatten()), dim=-1).to(device)

likelihoods = []

for i in range(num_trajectories):
    with torch.no_grad():
        x = input[i].unsqueeze(0)
        x_t = x[:, -1:, :]
        y = target[i].unsqueeze(0)
        feat = feature[i].unsqueeze(0)

        x, y, angle = traj_flow._normalize_rotation(x, y)
        feat = traj_flow._rotate_features(feat, angle)
        embedding = traj_flow._embedding(x, feat)
        embedding = embedding.repeat(batch_size, 1)

        pz_t1 = []
        for grid_batch in grid.split(batch_size, dim=0):
            grid_batch = grid_batch.unsqueeze(1).expand(-1, 12, -1)
            grid_batch = traj_flow._rotate(grid_batch, x_t, angle)
            z_t0, delta_logpz = traj_flow.flow(grid_batch, embedding)
            logpz_t0, logpz_t1 = traj_flow.log_prob(z_t0, delta_logpz)
            pz_t1.append(logpz_t1)
        
        pz_t1 = torch.cat(pz_t1, dim=0)
        likelihoods.append(pz_t1)
t = likelihoods[0]

grid_numpy = grid.cpu().detach().numpy()
xx = grid_numpy[:, 0].reshape(steps, steps)
yy = -grid_numpy[:, 1].reshape(steps, steps)
#likelihood = log_likelihood.exp().cpu().numpy().reshape(steps, steps)
#likelihood = t[:, 0].cpu().numpy().reshape(steps, steps)
likelihood = torch.mean(t, dim=-1).cpu().numpy().reshape(steps, steps)
likelihood = likelihood / np.max(likelihood)
plt.pcolormesh(xx, yy, likelihood, shading='auto', cmap=plt.cm.viridis, alpha=0.5, vmin=0, vmax=1)

plt.show()
