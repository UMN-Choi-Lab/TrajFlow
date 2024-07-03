import os
import subprocess
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from InD import InD # TODO: we should just extend torch dataloader instead
from TrajCNF import TrajCNF

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train parameters
train = False#True
verbose = True
epochs = 100
lr = 1e-3
scheduler_gamma = 0.999

# visualize parameters
visualize = True
num_samples_to_viz = 1
steps = 10#300
viz_batch_size = 10
output_dir = 'frames'

# data loader
# I need to easily get the background image associated with test set data
ind = InD(
    root="data",
    train_ratio=0.7, 
    train_batch_size=64, 
    test_batch_size=1)

# model
traj_cnf = TrajCNF(
    seq_len=100, 
    input_dim=2, 
    feature_dim=5, 
    embedding_dim=128).to(device)

# cnf train
if train:
    traj_cnf.train()

    optim = torch.optim.Adam(traj_cnf.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, scheduler_gamma)
    
    total_loss = []
    for epoch in range(epochs):
        losses = []
        for inputs, features in ind.observation_site8.train_loader:
            input = inputs[:, :100, ...].to(device)
            target = inputs[:, 100:, ...].to(device)
            features = features[:, :100, ...].to(device)

            z_t0, delta_logpz = traj_cnf(input, target, features)
            logpz_t0, logpz_t1 = traj_cnf.log_prob(z_t0, delta_logpz)
            loss = -torch.mean(logpz_t1)

            if verbose:
                print(f'logpz_t0 (latent): {-torch.mean(logpz_t0)}')
                print(f'logpz_t1 (prior): {loss}')

            total_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            losses.append(loss.item())
        
        print(f"epoch: {epoch}, loss: {np.mean(losses):.4f}")

    plt.plot(total_loss)

    torch.save(traj_cnf.state_dict(), 'traj_cnf.pt')

#traj_cnf.load_state_dict(torch.load('traj_cnf.pt'))

# cnf viz
if visualize:
    os.makedirs(output_dir, exist_ok=True)

    traj_cnf.eval()

    for i in range(1):
        inputs, features = next(iter(ind.observation_site8.test_loader))
        input = inputs[:, :100, ...].to(device)
        target = inputs[:, 100:, ...].to(device)
        features = features[:, :100, ...].to(device)

        linspace = torch.linspace(0, 1, steps)
        x, y = torch.meshgrid(linspace, linspace)
        grid = torch.stack((x.flatten(), y.flatten()), dim=-1).to(device)
    
        with torch.no_grad():
            embedding = traj_cnf._embedding(input, features)
            embedding = embedding.repeat(viz_batch_size, 1)

            pz_t1 = []
            for grid_batch in grid.split(viz_batch_size, dim=0):
                grid_batch = grid_batch.unsqueeze(1).expand(-1, 100, -1)
                z_t0, delta_logpz = traj_cnf.flow(grid_batch, embedding)
                logpz_t0, logpz_t1 = traj_cnf.log_prob(z_t0, delta_logpz)
                pz_t1.append(logpz_t1.exp())
        
            pz_t1 = torch.cat(pz_t1, dim=0)

    fudge_factor = 11.5
    ortho_px_to_meter = ind.observation_site8.ortho_px_to_meter * fudge_factor

    denormalized_grid = ind.observation_site8.denormalize(grid.cpu().numpy())
    x = denormalized_grid[:, 0].reshape(steps, steps)
    y = -denormalized_grid[:, 1].reshape(steps, steps)

    background = plt.imread('data/08_background.png')

    min_x = 0
    max_x = background.shape[1] * ortho_px_to_meter
    min_y = background.shape[0] * ortho_px_to_meter
    max_y = 0

    for t in range(100):
        likelihood = pz_t1[:, t].cpu().numpy().reshape(steps, steps)
        likelihood = likelihood / np.max(likelihood)
        likelihood = np.where(likelihood < 0.001, np.nan, likelihood)

        plt.figure(figsize=(10, 8))

        plt.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect='equal')

        color_map = plt.cm.viridis
        color_map.set_bad(color='none')
        heat_map = plt.pcolormesh(x, y, likelihood, shading='auto', cmap=color_map, alpha=0.5, vmin=0, vmax=1)
        plt.colorbar(heat_map, label='Likelihood')

        observed_traj = ind.observation_site8.denormalize(input[0].cpu().numpy())
        observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)
        plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='red', linewidth=1, label='Observed Trajectory')

        unobserved_traj = ind.observation_site8.denormalize(target[0].cpu().numpy())
        unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)
        plt.plot(unobserved_traj[:, 0], unobserved_traj[:, 1], color='lightcoral', linewidth=1, label='Unobserved Trajectory')
    
        plt.title(f'Density Heatmap Time: {t}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        plt.savefig(os.path.join(output_dir, f'frame_{t:03d}.png'))
        plt.close()

    command = ['ffmpeg', '-r', '10', '-i', 'frames/frame_%03d.png', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 'video5.mp4']
    subprocess.run(command, check=True)
