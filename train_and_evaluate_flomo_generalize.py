import torch
import torch.nn.functional as F
import numpy as np
from datasets.EthUcy import EthUcy
from datasets.InD import InD
from model.FloMo import FloMo
from tqdm import tqdm

def derivative_of(x, dt=1):
	not_nan_mask = ~np.isnan(x)
	masked_x = x[not_nan_mask]

	if masked_x.shape[-1] < 2:
		return np.zeros_like(x)

	dx = np.full_like(x, np.nan)
	dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

	return dx

def min_ade(y_true, y_pred):
    distances = torch.norm(y_pred - y_true.expand_as(y_pred), dim=-1)
    min_distances = torch.min(distances, dim=0).values
    return torch.mean(min_distances)

def min_fde(y_true, y_pred):
    fde = torch.norm(y_pred[:,-1,:] - y_true[:,-1,:], dim=-1)
    return fde.min()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ethucy = EthUcy(train_batch_size=128, test_batch_size=1, history=8, futures=12, smin=0.3, smax=1.7)
observation_site = ethucy.zara2_observation_site
flomo = FloMo(hist_size=8, pred_steps=12, alpha=10, beta=0.2, gamma=0.02, num_in=2, num_feat=0).to(device)

flomo.train()

optim = torch.optim.Adam(flomo.parameters(), lr=1e-3, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.999)

for epoch in range(25):
    losses = []
    for input, _, target in (pbar := tqdm(observation_site.train_loader)):
        input = input.to(device)
        target = target.to(device)

        log_prob = flomo.log_prob(target, input)
        loss = -torch.mean(log_prob)
            
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
            
        losses.append(loss)

        pbar.set_description(f'Epoch {epoch} Loss {loss.item():.4f}')

    losses = torch.stack(losses)
    pbar.set_description(f'Epoch {epoch} Loss {torch.mean(losses):.4f}')

ethucy = EthUcy(train_batch_size=128, test_batch_size=1, history=8, futures=24, smin=0.3, smax=1.7)
observation_site = ethucy.zara2_observation_site

flomo.eval()

with torch.no_grad():
    min_ade_sum = 0
    min_fde_sum = 0
    count = 0

    for test_input, test_feature, test_target in observation_site.test_loader:
        test_input = test_input.to(device)
        test_feature = test_feature.to(device)
        test_target = test_target.to(device)

        # sample based evaluation
        first_12, _ = flomo.sample(20, test_input)
        generalize = []
        for sample in first_12:
            last_8 = sample[4:, :].unsqueeze(0)
            next_12, _ = flomo.sample(1, last_8)
            generalize.append(next_12)

        next_12 = torch.cat(generalize, dim=0)
        samples = torch.cat((first_12, next_12), dim=1)

        test_target = torch.tensor(observation_site.denormalize(test_target.cpu().numpy())).to(device)
        samples = torch.tensor(observation_site.denormalize(samples.cpu().numpy())).to(device)

        min_ade_sum += min_ade(test_target, samples)
        min_fde_sum += min_fde(test_target, samples)
        count += 1

    print(f'generalized min ade: {min_ade_sum / count}')
    print(f'generalized min fde: {min_fde_sum / count}')