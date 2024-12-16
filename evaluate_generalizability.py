import torch
import numpy as np
from datasets.EthUcy import EthUcy

def derivative_of(x, dt=1):
	not_nan_mask = ~np.isnan(x)
	masked_x = x[not_nan_mask]

	if masked_x.shape[-1] < 2:
		return np.zeros_like(x)

	dx = np.full_like(x, np.nan)
	dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

	return dx

def append_time(features):
	seq_len, _ = features.shape
	t = torch.linspace(0., 1., seq_len)
	t = t.unsqueeze(-1)
	t = t.expand(seq_len, 1)
	return torch.cat([features, t], dim=-1)

def min_ade(y_true, y_pred):
	distances = torch.norm(y_pred - y_true.expand_as(y_pred), dim=-1)
	min_distances = torch.min(distances, dim=0).values
	return torch.mean(min_distances)

def min_fde(y_true, y_pred):
	fde = torch.norm(y_pred[:,-1,:] - y_true[:,-1,:], dim=-1)
	return fde.min()

def evaluate_generalizability(observation_site_name, model, num_samples, device):
	model.eval()

	ethucy = EthUcy(train_batch_size=128, test_batch_size=1, history=8, futures=24, smin=0.3, smax=1.7, relaxed=False)
	observation_site = (
		ethucy.eth_observation_site if observation_site_name == 'eth' else
		ethucy.hotel_observation_site if observation_site_name == 'hotel' else
		ethucy.univ_observation_site if observation_site_name == 'univ' else
		ethucy.zara1_observation_site if observation_site_name == 'zara1' else
		ethucy.zara2_observation_site if observation_site_name == 'zara2' else
		ethucy.zara2_observation_site
	)

	with torch.no_grad():
		min_ade_sum = 0
		min_fde_sum = 0
		count = 0

		for test_input, test_feature, test_target in observation_site.test_loader:
			test_input = test_input.to(device)
			test_feature = test_feature.to(device)
			test_target = test_target.to(device)

			if model.marginal:            
				_, samples, _ = model.sample(test_input, test_feature, 24, num_samples)
			else:
				_, first_12, _ = model.sample(test_input, test_feature, 12, num_samples)
				generalize = []
				for sample in first_12:
					x = sample[:, 0].cpu().numpy()
					y = sample[:, 1].cpu().numpy()
					#x = last_8[:, 0].cpu().numpy()
					#y = last_8[:, 1].cpu().numpy()

					dt = 0.4
					vx = derivative_of(x, dt)
					vy = derivative_of(y, dt)
					ax = derivative_of(vx, dt)
					ay = derivative_of(vy, dt)

					last_8 = sample[4:, :]

					vx = torch.tensor(vx)[4:].unsqueeze(-1)
					vy = torch.tensor(vy)[4:].unsqueeze(-1)
					ax = torch.tensor(ax)[4:].unsqueeze(-1)
					ay = torch.tensor(ay)[4:].unsqueeze(-1)
					last_8_features = torch.cat((vx, vy, ax, ay), dim=-1)
					last_8_features = append_time(last_8_features).to(device)

					_, next_12, _ = model.sample(last_8.unsqueeze(0), last_8_features.unsqueeze(0), 12, 1)
					generalize.append(next_12)

				next_12 = torch.cat(generalize, dim=0)
				samples = torch.cat((first_12, next_12), dim=1)

			test_target = torch.tensor(observation_site.denormalize(test_target.cpu().numpy())).to(device)
			samples = torch.tensor(observation_site.denormalize(samples.cpu().numpy())).to(device)
			
			samples = samples[:,:test_target.shape[1],:]
			min_ade_sum += min_ade(test_target, samples)
			min_fde_sum += min_fde(test_target, samples)
			count += 1

		min_ade_score = min_ade_sum / count
		min_fde_score = min_fde_sum / count

	return min_ade_score, min_fde_score