import torch
import matplotlib.pyplot as plt
import numpy as np
from TrajCNF import TrajCNF

# hyper parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
num_samples_to_viz = 1
steps = 300
test_batch_size = 10

# model
traj_flow = TrajCNF(seq_len=seq_len, input_dim=input_dim, feature_dim=feature_dim, embedding_dim=embedding_dim).to(device)

# data manager
# TODO: need to port data manager from other repo. 
# I need to easily get the background image associated with test set data

# cnf train
traj_flow.train()
if train:
   optim = torch.optim.Adam(traj_flow.parameters(), lr=lr)
   scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, scheduler_gamma)
   total_loss = []
   for epoch in range(epochs):
       losses = []
       for inputs, features in dm.train_loader:
           input = inputs[:, :100, ...].to(device)
           target = inputs[:, 100:, ...].to(device)
           features = features[:, :100, ...].to(device)

           z_t0, delta_logpz = traj_flow(input, target, features)
           logpz_t0, logpz_t1 = traj_flow.log_prob(z_t0, delta_logpz)
           loss = -torch.mean(logpz_t1)

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

   torch.save(traj_flow.state_dict(), 'traj_cnf.pt')

traj_flow.load_state_dict(torch.load('traj_cnf.pt'))

# cnf viz
# TODO: need to overlay on background image and use ffmpeg to make a video
traj_flow.eval()
for i in range(num_samples_to_viz):
    inputs, features = next(iter(dm.test_loader))
    input = inputs[:, :100, ...].to(device)
    target = inputs[:, 100:, ...].to(device)
    features = features[:, :100, ...].to(device)

    linspace = torch.linspace(0, 1, steps)
    x, y = torch.meshgrid(linspace, linspace)
    grid = torch.stack((x.flatten(), y.flatten()), dim=-1).to(device)
    
    with torch.no_grad():
        embedding = traj_flow._embedding(input, features)
        embedding = embedding.repeat(test_batch_size, 1)

        pz_t1 = []
        for grid_batch in grid.split(test_batch_size, dim=0):
            grid_batch = grid_batch.unsqueeze(1).expand(-1, 100, -1)
            z_t0, delta_logpz = traj_flow.flow(grid_batch, embedding)
            logpz_t0, logpz_t1 = traj_flow.log_prob(z_t0, delta_logpz)
            pz_t1.append(logpz_t1.exp())
        
        pz_t1 = torch.cat(pz_t1, dim=0)

    x = grid[:, 0].cpu().numpy().reshape(steps, steps)
    y = grid[:, 1].cpu().numpy().reshape(steps, steps)
    likelihood = pz_t1[:,0].cpu().numpy().reshape(steps, steps)

    plt.figure(figsize=(10, 8))

    heatmap = plt.pcolormesh(x, y, likelihood, shading='auto', cmap='viridis')
    plt.colorbar(heatmap, label='Likelihood')
    
    input_cpu = input[0].cpu().numpy()
    plt.plot(input_cpu[:, 0], input_cpu[:, 1], color='red', linewidth=2, label='Input')
    
    target_cpu = target[0].cpu().numpy()
    plt.plot(target_cpu[:, 0], target_cpu[:, 1], color='blue', linewidth=2, label='Target')
    
    plt.title('Density Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.show()