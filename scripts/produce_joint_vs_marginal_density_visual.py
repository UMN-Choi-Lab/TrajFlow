# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# from PIL import Image
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from datasets.EthUcy import EthUcy
# from model.TrajFlow import TrajFlow, CausalEnocder, Flow

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# eth = EthUcy(train_batch_size=128, test_batch_size=1, history=8, futures=12)
# observation_site = eth.zara1_observation_site

# traj_flow_j = TrajFlow(
#     seq_len=12, input_dim=2, feature_dim=4, 
#     embedding_dim=128, hidden_dim=512, 
#     causal_encoder=CausalEnocder.CDE,
#     flow=Flow.CNF,
#     marginal=False).to(device)
# traj_flow_j.load_state_dict(torch.load('ind_joint.pt'))
# traj_flow_j.eval()

# traj_flow_m = TrajFlow(
#     seq_len=12, input_dim=2, feature_dim=4,
#     embedding_dim=128, hidden_dim=512,
#     causal_encoder=CausalEnocder.CDE,
#     flow=Flow.CNF,
#     marginal=True).to(device)
# traj_flow_m.load_state_dict(torch.load('ind_marginal.pt'))
# traj_flow_m.eval()

# data = list(observation_site.test_loader)
# input, feature, target = data[0]
# input = input.to(device)
# feature = feature.to(device)
# target = target.to(device)

# observed_traj = input[0].cpu().numpy()
# observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)

# unobserved_traj = target[0].cpu().numpy()
# unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)

# z_t0, samples, delta_logpz = traj_flow_j.sample(input, feature, 12, 20)
# logpz_t0, logpz_t1 = traj_flow_j.log_prob(z_t0, delta_logpz)
# logpz_t1 = -logpz_t1
# min_val = logpz_t1.min()
# max_val = logpz_t1.max()
# logpz_t1 = (logpz_t1 - min_val) / (max_val - min_val)

# linewidth = 5
# color_map = plt.cm.viridis

# # joint density visual
# plt.figure(figsize=(10, 8))
# plt.axis('off')
# plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='black', linewidth=linewidth, label='Observed Trajectory')

# last_observed_point = observed_traj[-1]
# x_center, y_center = last_observed_point[0], last_observed_point[1]
# x_range = 0.3
# y_range = 0.3
# plt.xlim(x_center - x_range, x_center + x_range)
# plt.ylim(y_center - y_range, y_center + y_range)

# likelihoods = logpz_t1.cpu().detach().numpy()

# for i in np.argsort(likelihoods):
#     sampled_traj = samples[i].cpu().detach().numpy()
#     sampled_traj = np.stack([sampled_traj[:, 0], -sampled_traj[:, 1]], axis=-1)
#     likelihood = likelihoods[i]
#     color = color_map(likelihood)
#     plt.plot([last_observed_point[0], sampled_traj[0, 0]], [last_observed_point[1], sampled_traj[0, 1]], color=color, linewidth=linewidth)
#     plt.plot(sampled_traj[:, 0], sampled_traj[:, 1], color=color, linewidth=linewidth, label='Sampled Trajectory')

# joint_forecast_file_name = 'joint_forecast.png'
# plt.savefig(joint_forecast_file_name, bbox_inches='tight')
# plt.show()

# #marginal density visual
# steps = 500
# linspace = torch.linspace(0, 1, steps)
# x, y = torch.meshgrid(linspace, linspace)
# grid = torch.stack((x.flatten(), y.flatten()), dim=-1).to(device)

# with torch.no_grad():
#     batch_size = 1000
#     embedding = traj_flow_m._embedding(input, feature)
#     embedding = embedding.repeat(batch_size, 1)

#     pz_t1 = []
#     for grid_batch in grid.split(batch_size, dim=0):
#         grid_batch = grid_batch.unsqueeze(1).expand(-1, 12, -1)
#         z_t0, delta_logpz = traj_flow_m.flow(grid_batch, embedding)
#         logpz_t0, logpz_t1 = traj_flow_m.log_prob(z_t0, delta_logpz)
#         pz_t1.append(logpz_t1.exp())
        
#     pz_t1 = torch.cat(pz_t1, dim=0)

# plt.figure(figsize=(10, 8))
# plt.axis('off')
# plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='black', linewidth=linewidth, label='Observed Trajectory')

# plt.xlim(x_center - x_range, x_center + x_range)
# plt.ylim(y_center - y_range, y_center + y_range)

# grid = grid.cpu().detach().numpy()
# x = grid[:, 0].reshape(steps, steps)
# y = -grid[:, 1].reshape(steps, steps)
# for t in [0, 1, 2, 3, 5, 7, 11]:
#     likelihood = pz_t1[:, t].cpu().numpy().reshape(steps, steps)
#     likelihood = likelihood / np.max(likelihood)
#     likelihood = np.where(likelihood < 0.25, np.nan, likelihood)
#     plt.pcolormesh(x, y, likelihood, shading='auto', cmap=color_map, alpha=0.5, vmin=0, vmax=1)

# marginal_forecast_file_name = 'marginal_forecast.png'
# plt.savefig(marginal_forecast_file_name, bbox_inches='tight')
# plt.show()

# joint_forecast_img = Image.open(joint_forecast_file_name)
# marginal_forecast_img = Image.open(marginal_forecast_file_name)

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0].imshow(joint_forecast_img)
# axes[0].axis('off')

# axes[1].imshow(marginal_forecast_img)
# axes[1].axis('off')

# axes[0].text(0.5, -0.1, 'a) Joint distribution', ha='center', va='top', transform=axes[0].transAxes, fontsize=14)
# axes[1].text(0.5, -0.1, 'b) Marginal distribution', ha='center', va='top', transform=axes[1].transAxes, fontsize=14)

# handles = [plt.Line2D([0], [0], color='black', lw=4, label='Observed Trajectory')]
# fig.legend(handles=handles)

# norm = mcolors.Normalize(vmin=0, vmax=1)
# dummy_mappable = cm.ScalarMappable(norm=norm, cmap=color_map)
# dummy_mappable.set_array([])
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
# cbar = plt.colorbar(dummy_mappable, cax=cbar_ax, label='Likelihood')

# #plt.subplots_adjust(left=0, right=1)

# plt.savefig('forecasts.png')
# plt.show()

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.EthUcy import EthUcy
from model.TrajFlow import TrajFlow, CausalEnocder, Flow

device = 'cuda' if torch.cuda.is_available() else 'cpu'

eth = EthUcy(train_batch_size=128, test_batch_size=1, history=8, futures=12)
observation_site = eth.zara1_observation_site

traj_flow_j = TrajFlow(
    seq_len=12, input_dim=2, feature_dim=4, 
    embedding_dim=128, hidden_dim=512, 
    causal_encoder=CausalEnocder.CDE,
    flow=Flow.CNF,
    marginal=False).to(device)
traj_flow_j.load_state_dict(torch.load('ethucy_joint.pt'))
traj_flow_j.eval()

traj_flow_m = TrajFlow(
    seq_len=12, input_dim=2, feature_dim=4,
    embedding_dim=128, hidden_dim=512,
    causal_encoder=CausalEnocder.CDE,
    flow=Flow.CNF,
    marginal=True).to(device)
traj_flow_m.load_state_dict(torch.load('ethucy_marginal.pt'))
traj_flow_m.eval()

data = list(observation_site.test_loader)
input, feature, target = data[0]
input = input.to(device)
feature = feature.to(device)
target = target.to(device)

observed_traj = input[0].cpu().numpy()
observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)

unobserved_traj = target[0].cpu().numpy()
unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)

z_t0, samples, delta_logpz = traj_flow_j.sample(input, feature, 12, 20)
logpz_t0, logpz_t1 = traj_flow_j.log_prob(z_t0, delta_logpz)
logpz_t1 = -logpz_t1
min_val = logpz_t1.min()
max_val = logpz_t1.max()
logpz_t1 = (logpz_t1 - min_val) / (max_val - min_val)

linewidth = 5
color_map = plt.cm.viridis

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].axis('off')
axes[0].plot(observed_traj[:, 0], observed_traj[:, 1], color='black', linewidth=linewidth, label='Observed Trajectory')

last_observed_point = observed_traj[-1]
x_center, y_center = last_observed_point[0], last_observed_point[1]
x_range = 0.3
y_range = 0.3
axes[0].set_xlim(x_center - x_range, x_center + x_range)
axes[0].set_ylim(y_center - y_range, y_center + y_range)

likelihoods = logpz_t1.cpu().detach().numpy()
for i in np.argsort(likelihoods):
    sampled_traj = samples[i].cpu().detach().numpy()
    sampled_traj = np.stack([sampled_traj[:, 0], -sampled_traj[:, 1]], axis=-1)
    likelihood = likelihoods[i]
    color = color_map(likelihood)
    axes[0].plot([last_observed_point[0], sampled_traj[0, 0]], [last_observed_point[1], sampled_traj[0, 1]], color=color, linewidth=linewidth)
    axes[0].plot(sampled_traj[:, 0], sampled_traj[:, 1], color=color, linewidth=linewidth)

axes[1].axis('off')
axes[1].plot(observed_traj[:, 0], observed_traj[:, 1], color='black', linewidth=linewidth, label='Observed Trajectory')

axes[1].set_xlim(x_center - x_range, x_center + x_range)
axes[1].set_ylim(y_center - y_range, y_center + y_range)

steps = 1000
linspace = torch.linspace(0, 1, steps)
x, y = torch.meshgrid(linspace, linspace)
grid = torch.stack((x.flatten(), y.flatten()), dim=-1).to(device)

with torch.no_grad():
    batch_size = 1000
    embedding = traj_flow_m._embedding(input, feature)
    embedding = embedding.repeat(batch_size, 1)

    pz_t1 = []
    for grid_batch in grid.split(batch_size, dim=0):
        grid_batch = grid_batch.unsqueeze(1).expand(-1, 12, -1)
        z_t0, delta_logpz = traj_flow_m.flow(grid_batch, embedding)
        logpz_t0, logpz_t1 = traj_flow_m.log_prob(z_t0, delta_logpz)
        pz_t1.append(logpz_t1.exp())

    pz_t1 = torch.cat(pz_t1, dim=0)

grid = grid.cpu().detach().numpy()
x = grid[:, 0].reshape(steps, steps)
y = -grid[:, 1].reshape(steps, steps)

for t in [0, 1, 2, 3, 5, 7, 11]:
    likelihood = pz_t1[:, t].cpu().numpy().reshape(steps, steps)
    likelihood = likelihood / np.max(likelihood)
    likelihood = np.where(likelihood < 0.25, np.nan, likelihood)
    axes[1].pcolormesh(x, y, likelihood, shading='auto', cmap=color_map, alpha=0.5, vmin=0, vmax=1)

axes[0].text(0.5, -0.1, 'a) Joint distribution', ha='center', va='top', transform=axes[0].transAxes, fontsize=18)
axes[1].text(0.5, -0.1, 'b) Marginal distribution', ha='center', va='top', transform=axes[1].transAxes, fontsize=18)

handles = [plt.Line2D([0], [0], color='black', lw=4, label='Observed Trajectory')]
fig.legend(handles=handles, prop={'size': 12})

norm = mcolors.Normalize(vmin=0, vmax=1)
dummy_mappable = cm.ScalarMappable(norm=norm, cmap=color_map)
dummy_mappable.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(dummy_mappable, cax=cbar_ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Likelihood', fontsize=14)

plt.savefig('forecasts.pdf', format='pdf')
plt.show()
