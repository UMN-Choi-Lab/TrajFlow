import os
import subprocess
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TrajDataLoader import load_data # TODO: we should just extend torch dataloader instead
from TrajCNF import TrajCNF

# hyper parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
verbose = True

# data loader parameters
train_ratio = 0.7
train_batch_size = 64
test_batch_size = 1

# model parameters
seq_len = 100
input_dim = 2
feature_dim = 5
embedding_dim = 128

# train parameters
train = True
epochs = 100
lr = 1e-3
scheduler_gamma = 0.999

# visualize parameters
visualize = True
num_samples_to_viz = 1
steps = 300
viz_batch_size = 10
output_dir = 'frames'

# data loader
# I need to easily get the background image associated with test set data
input = None
features = None
train_loader, test_loader = load_data(
    input=input, features=features, 
    train_ratio=train_ratio, 
    train_batch_size=train_batch_size, 
    test_batch_size=test_batch_size)

# model
traj_cnf = TrajCNF(
    seq_len=seq_len, 
    input_dim=input_dim, 
    feature_dim=feature_dim, 
    embedding_dim=embedding_dim).to(device)

# cnf train
if train:
    traj_cnf.train()

    optim = torch.optim.Adam(traj_cnf.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, scheduler_gamma)
    
    total_loss = []
    for epoch in range(epochs):
        losses = []
        for inputs, features in train_loader:
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

traj_cnf.load_state_dict(torch.load('traj_cnf.pt'))

# cnf viz
def denormalization(raw_input, x_min, x_max, y_min, y_max):
    # x_min = 30
    # x_max = 81
    # y_min = -61
    # y_max = -5

    def de_normalize_x(x): return x*(x_max-x_min)+x_min if x != -1 else -1
    def de_normalize_y(y): return y*(y_max-y_min)+y_min if y != -1 else -1

    raw_input[:, :, 0] = np.vectorize(de_normalize_x)(raw_input[:, :, 0])
    raw_input[:, :, 1] = np.vectorize(de_normalize_y)(raw_input[:, :, 1])
    return raw_input

# TODO: need to overlay on background image and use ffmpeg to make a video
if visualize:
    os.makedirs(output_dir, exist_ok=True)

    traj_cnf.eval()

    for i in range(1):
        inputs, features = next(iter(test_loader))
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

        metadata = pd.read_csv('data/08_recordingMeta.csv').to_dict(orient="records")[0]
        ortho_px_to_meter = metadata["orthoPxToMeter"] * 11.5 # why 12?

    denormalized_grid = denormalization(grid.unsqueeze(0).cpu().numpy(), 30, 81, -61, -1)[0]
    x = denormalized_grid[:, 0].reshape(steps, steps)
    y = -denormalized_grid[:, 1].reshape(steps, steps)

    for t in range(100):
        likelihood = pz_t1[:, t].cpu().numpy().reshape(steps, steps)

        plt.figure(figsize=(10, 8))

        background = plt.imread('data/08_background.png')

        min_x = 0
        max_x = background.shape[1] * ortho_px_to_meter
        min_y = background.shape[0] * ortho_px_to_meter
        max_y = 0

        plt.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect='equal')

        heatmap = plt.pcolormesh(x, y, likelihood, shading='auto', cmap='viridis', alpha=0.5)
        plt.colorbar(heatmap, label='Likelihood')

        observed_traj = denormalization(input.cpu().numpy(), 30, 81, -61, -1)[0]
        observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)
        plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='red', linewidth=2, label='Observed Trajectory')

        unobserved_traj = denormalization(target.cpu().numpy(), 30, 81, -61, -1)[0]
        unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)
        plt.plot(unobserved_traj[:, 0], unobserved_traj[:, 1], color='lightcoral', linewidth=2, label='Unobserved Trajectory')
    
        plt.title('Density Heatmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        plt.savefig(os.path.join(output_dir, f'frame_{t:03d}.png'))
        plt.close()

    command = ['ffmpeg', '-r', '10', '-i', 'frames/frame_%03d.png', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 'video5.mp4']
    subprocess.run(command, check=True)
