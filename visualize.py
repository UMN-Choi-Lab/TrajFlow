import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_pzt1(model, input, features, grid):
    with torch.no_grad():
        batch_size = 10

        embedding = model._embedding(input, features)
        embedding = embedding.repeat(batch_size, 1)

        pz_t1 = []
        for grid_batch in grid.split(batch_size, dim=0):
            grid_batch = grid_batch.unsqueeze(1).expand(-1, 100, -1)
            z_t0, delta_logpz = model.flow(grid_batch, embedding)
            logpz_t0, logpz_t1 = model.log_prob(z_t0, delta_logpz)
            pz_t1.append(logpz_t1.exp())
        
        pz_t1 = torch.cat(pz_t1, dim=0)
        return pz_t1
    
def generate_video(background_image, grid, pz_t1, observed_traj, unobserved_traj, 
                   ortho_px_to_meter, steps, output_dir, i):
    frames_dir = f'{output_dir}/video{i}'
    os.makedirs(frames_dir, exist_ok=True)

    x = grid[:, 0].reshape(steps, steps)
    y = -grid[:, 1].reshape(steps, steps)

    background = plt.imread(background_image)

    min_x = 0
    max_x = background.shape[1] * ortho_px_to_meter
    min_y = background.shape[0] * ortho_px_to_meter
    max_y = 0

    for t in range(100):
        likelihood = pz_t1[:, t].cpu().numpy().reshape(steps, steps)
        likelihood = likelihood / np.max(likelihood)
        likelihood = np.where(likelihood < 0.001, np.nan, likelihood)
        generate_frame(background, x, y, likelihood, observed_traj, unobserved_traj,
                        min_x, max_x, min_y, max_y, t, output_dir)

    command = ['ffmpeg', '-r', '10', '-i', f'{frames_dir}/frame_%03d.png', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'video{i}.mp4']
    subprocess.run(command, check=True)

    
def generate_frame(background, x, y, likelihood, observed_traj, unobserved_traj,
                   min_x, max_x, min_y, max_y, t, output_dir):
    plt.figure(figsize=(10, 8))
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect='equal')

    color_map = plt.cm.viridis
    color_map.set_bad(color='none')
    heat_map = plt.pcolormesh(x, y, likelihood, shading='auto', cmap=color_map, alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(heat_map, label='Likelihood')
    
    plt.plot(observed_traj[:, 0], observed_traj[:, 1], color='red', linewidth=1, label='Observed Trajectory')
    plt.plot(unobserved_traj[:, 0], unobserved_traj[:, 1], color='lightcoral', linewidth=1, label='Unobserved Trajectory')
    
    plt.title(f'Density Heatmap Time: {t}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.savefig(os.path.join(output_dir, f'frame_{t:03d}.png'))
    plt.close()

def visualize(observation_site, model, num_samples, steps, output_dir):
    if os.path.exists(output_dir):
        os.removedirs(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    fudge_factor = 11.5
    ortho_px_to_meter = observation_site.ortho_px_to_meter * fudge_factor

    for i in range(num_samples):
        inputs, features = next(iter(observation_site.test_loader))
        input = inputs[:, :100, ...].to(device)
        target = inputs[:, 100:, ...].to(device)
        features = features[:, :100, ...].to(device)

        linspace = torch.linspace(0, 1, steps)
        x, y = torch.meshgrid(linspace, linspace)
        grid = torch.stack((x.flatten(), y.flatten()), dim=-1).to(device)
    
        pz_t1 = compute_pzt1(model, input, features, grid)

        observed_traj = observation_site.denormalize(input[0].cpu().numpy())
        observed_traj = np.stack([observed_traj[:, 0], -observed_traj[:, 1]], axis=-1)

        unobserved_traj = observation_site.denormalize(target[0].cpu().numpy())
        unobserved_traj = np.stack([unobserved_traj[:, 0], -unobserved_traj[:, 1]], axis=-1)

        denormalized_grid = observation_site.denormalize(grid.cpu().numpy())

        generate_video(observation_site.background, denormalized_grid, pz_t1,
                       observed_traj, unobserved_traj, ortho_px_to_meter,
                       steps, output_dir, i)

        