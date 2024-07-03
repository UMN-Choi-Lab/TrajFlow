
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(observation_site, model, epochs, lr, gamma, verbose):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
    
    total_loss = []
    for epoch in range(epochs):
        losses = []
        for inputs, features in observation_site.train_loader:
            input = inputs[:, :100, ...].to(device)
            target = inputs[:, 100:, ...].to(device)
            features = features[:, :100, ...].to(device)

            z_t0, delta_logpz = model(input, target, features)
            logpz_t0, logpz_t1 = model.log_prob(z_t0, delta_logpz)
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
        
        losses = torch.stack(losses)
        print(f"epoch: {epoch}, loss: {torch.mean(losses):.4f}")

    #plt.plot(total_loss)

    torch.save(model.state_dict(), 'traj_cnf.pt')