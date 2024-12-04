import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Scene():
	def __init__(self):
		self.agents = []

class Agent():
	def __init__(self, trajectory):
		self.trajectory = trajectory

class EthUcyDataset(Dataset):
	def __init__(self, scenes, history_frames=8, future_frames=12, smin=0.3, smax=1.7, evaluation_mode=False):
		self.scenes = scenes
		self.history_frames = history_frames
		self.future_frames = future_frames
		self.smin = smin
		self.smax = smax
		self.evaluation_mode = evaluation_mode
		self.data = self._prepare_data()

	def _prepare_data(self):
		data = []
		for scene in self.scenes:
			for agent in scene.agents:
				if self.evaluation_mode:
					if len(agent.trajectory) >= self.history_frames + 2:
						history = agent.trajectory[0:self.history_frames]
						future = agent.trajectory[self.history_frames:self.history_frames+self.future_frames]
						data.append((history, future))
				else:
					for i in range(self.history_frames - 1, len(agent.trajectory) - self.future_frames):
						history = agent.trajectory[i-self.history_frames+1:i+1]
						future = agent.trajectory[i+1:i+1+self.future_frames]
						data.append((history, future))
		return data
	
	def _append_time(self, features):
		seq_len, _ = features.shape
		t = torch.linspace(0., 1., seq_len)
		t = t.unsqueeze(-1)
		t = t.expand(seq_len, 1)
		return torch.cat([features, t], dim=-1)
	
	def _augment_trajectories(self, history, future):
		full_trajectory = torch.cat([history, future], dim=0)
		mean_position = full_trajectory.mean(dim=0, keepdim=True)

		centered_history = history - mean_position
		centered_future = future - mean_position

		scaling_factors = torch.normal(mean=1.0, std=0.5, size=(1, 1))
		scaling_factors = torch.clamp(scaling_factors, min=self.smin, max=self.smax)

		scaled_history = centered_history * scaling_factors
		scaled_future = centered_future * scaling_factors

		augmented_history = scaled_history + mean_position
		augmented_future = scaled_future + mean_position

		return augmented_history, augmented_future
	
	def _derivative_of(self, x, dt=1):
		not_nan_mask = ~torch.isnan(x)
		masked_x = x[not_nan_mask]

		if masked_x.numel() < 2:
			return torch.zeros_like(x)

		dx = torch.full_like(x, float('nan'))
		dx[not_nan_mask] = torch.cat([torch.tensor([masked_x[1] - masked_x[0]]), torch.diff(masked_x)]) / dt

		return dx

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		history, future = self.data[idx]
		if not self.evaluation_mode:
			history, future = self._augment_trajectories(history, future)

		dt = 0.4
		x = history[:, 0]
		y = history[:, 1]
		vx = self._derivative_of(x, dt)
		vy = self._derivative_of(y, dt)
		ax = self._derivative_of(vx, dt)
		ay = self._derivative_of(vy, dt)

		features = torch.stack((vx, vy, ax, ay), dim=1)

		return history, features, future
	
class EthUcyObservationSite():
	def __init__(self, train_loader, test_loader):
		self.train_loader = train_loader
		self.test_loader = test_loader

	def normalize(self, data): # for compatability with inD
		return data
	
	def denormalize(self, data): # for compatability with inD
		return data

class EthUcy():
	def __init__(self, train_batch_size, test_batch_size, history, futures, smin, smax):
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.observation_sites = {}
		self.history = history
		self.futures = futures
		self.smin = smin
		self.smax = smax

	@property
	def eth_observation_site(self):
		return self._get_observation_site('eth')
	
	@property
	def hotel_observation_site(self):
		return self._get_observation_site('hotel')
	
	@property
	def univ_observation_site(self):
		return self._get_observation_site('univ')
	
	@property
	def zara1_observation_site(self):
		return self._get_observation_site('zara1')
	
	@property
	def zara2_observation_site(self):
		return self._get_observation_site('zara2')
	
	def _get_observation_site(self, data_source):
		if data_source not in self.observation_sites:
			train_scenes = self._load_data_source(data_source, 'train')
			test_scenes = self._load_data_source(data_source, 'test')
			train_loader = self._prepare_data(train_scenes, self.train_batch_size, False)
			test_loader = self._prepare_data(test_scenes, self.test_batch_size, True)
			self.observation_sites[data_source] = EthUcyObservationSite(train_loader, test_loader)
		return self.observation_sites[data_source]
	
	def _prepare_data(self, scenes, batch_size, evaluation_set):
		dataset = EthUcyDataset(scenes, self.history, self.futures, self.smin, self.smax, evaluation_set)
		return DataLoader(dataset, batch_size=batch_size, shuffle=True)

	def _load_data_source(self, data_source, data_class):
		scenes = []
		for subdir, _, files in os.walk(os.path.join('data', 'raw', data_source, data_class)):
			for file in files:
				if file.endswith('.txt'):
					full_data_path = os.path.join(subdir, file)
					print('At', full_data_path)

					data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
					data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
					data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
					data['frame_id'] = data['frame_id'] // 10
					data['frame_id'] -= data['frame_id'].min()
					data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

					data['node_id'] = data['track_id'].astype(str)
					data.sort_values('frame_id', inplace=True)

					data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
					data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

					scene = Scene()

					for node_id in pd.unique(data['node_id']):
						node_df = data[data['node_id'] == node_id]
						assert np.all(np.diff(node_df['frame_id']) == 1)

						node_values = node_df[['pos_x', 'pos_y']].values

						if node_values.shape[0] < 2:
							continue

						x = node_values[:, 0]
						y = node_values[:, 1]

						x_tensor = torch.from_numpy(x).float()
						y_tensor = torch.from_numpy(y).float()
						trajectory = torch.stack((x_tensor, y_tensor), dim=1)
						scene.agents.append(Agent(trajectory))
					
					scenes.append(scene)

		return scenes