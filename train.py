
import os
import torch
import matplotlib.pyplot as plt

def train(observation_site, model, epochs, lr, weight_decay, gamma, verbose, device):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
    
    total_loss = []
    for epoch in range(epochs):
        losses = []
        for input, feature, target in observation_site.train_loader:
            input = input.to(device)
            features = feature.to(device)
            target = target.to(device)

            z_t0, delta_logpz = model(input, target, features)
            logpz_t0, logpz_t1 = model.log_prob(z_t0, delta_logpz)
            loss = -torch.mean(logpz_t1)

            if verbose:
                print(f'logpz_t0 (latent): {-torch.mean(logpz_t0)}')
                print(f'logpz_t1 (prior): {loss}')
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            
            if verbose:
                total_loss.append(loss.item())
            losses.append(loss)

        losses = torch.stack(losses)
        epoch_loss = torch.mean(torch.mean(losses))
        if not verbose:
            total_loss.append(epoch_loss.item())
        print(f"epoch: {epoch}, loss: {epoch_loss:.4f}")


    if verbose:
        loss_visual = 'loss.png'

        if os.path.exists(loss_visual):
            os.remove(loss_visual)
        
        plt.plot(total_loss)
        plt.savefig(loss_visual)
        plt.close()

    return total_loss
    