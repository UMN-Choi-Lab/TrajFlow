import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt

# InD uses this to.... put in shared location
def normalize(data, boundaries):
	return (data - boundaries[:, 0]) / (boundaries[:, 1] - boundaries[:, 0])

def denormalize(data, boundaries):
	return (data * (boundaries[:, 1] - boundaries[:, 0])) + boundaries[:, 0]

def derivative_of(x, dt=1):
	not_nan_mask = ~np.isnan(x)
	masked_x = x[not_nan_mask]

	if masked_x.shape[-1] < 2:
		return np.zeros_like(x)

	dx = np.full_like(x, np.nan)
	dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

	return dx

class Scene():
	def __init__(self, timesteps):
		self.timesteps = timesteps
		self.agents = []

class Agent():
	def __init__(self, first_timestep, data):
		self.first_timestep = first_timestep
		self.data = data

class EthUcyDataset(Dataset):
	def __init__(self, scenes, history_frames=8, future_frames=12, evaluation_mode=False):
		self.scenes = scenes
		self.history_frames = history_frames
		self.future_frames = future_frames
		self.evaluation_mode = evaluation_mode
		self.data = self._prepare_data()

	def _prepare_data(self):
		data = []
		count = 0
		bcount = 0
		tcount = 0
		scount = 0
		for scene in self.scenes:
			for agent in scene.agents:
				count += 1
				if len(agent.data) >= 20:
					bcount += 1
				if len(agent.data) >= 9:
					tcount += 1
				if len(agent.data) > 2:
					scount += 1
				if self.evaluation_mode:
					if len(agent.data) > self.history_frames:
						history = agent.data.iloc[0:self.history_frames]
						future = agent.data.iloc[self.history_frames:self.history_frames+self.future_frames]
						data.append((history, future))
					elif len(agent.data) > 2:
						history = agent.data.iloc[:-1]
						future = agent.data.iloc[-1:]
						data.append((history, future))
				else:
					for i in range(self.history_frames - 1, len(agent.data) - self.future_frames):
						history = agent.data.iloc[i-self.history_frames+1:i+1]
						future = agent.data.iloc[i+1:i+1+self.future_frames]
						data.append((history, future))
		print(bcount / count)
		print(tcount / count)
		print(scount / count)
		return data
	
	def _append_time(self, features):
		seq_len, _ = features.shape
		t = torch.linspace(0., 1., seq_len)
		t = t.unsqueeze(-1)
		t = t.expand(seq_len, 1)
		return torch.cat([features, t], dim=-1)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		history, future = self.data[idx]

		input_columns = ['x_pos', 'y_pos']
		feature_columns = [col for col in history.columns if col not in input_columns]

		inputs = history[input_columns].values
		features = history[feature_columns].values
		targets = future[input_columns].values

		inputs = torch.tensor(inputs, dtype=torch.float32)
		features = torch.tensor(features, dtype=torch.float32)
		features = self._append_time(features)
		targets = torch.tensor(targets, dtype=torch.float32)

		return inputs, features, targets
	
class EthUcyObservationSite():
	def __init__(self, train_loader, test_loader, boundaries):
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.boundaries = boundaries

	def normalize(self, data):
		return normalize(data, self.boundaries)
	
	def denormalize(self, data):
		return denormalize(data, self.boundaries)

class EthUcy():
	def __init__(self, train_batch_size, test_batch_size, history, futures, min_futures):
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.observation_sites = {}
		self.history = history
		self.futures = futures
		self.min_futures = min_futures

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
			spatial_boundaries = np.array([[np.inf, -np.inf], [np.inf, -np.inf]])
			feature_boundaries = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]])
			train_scenes = self._load_data_source(data_source, 'train', spatial_boundaries, feature_boundaries)
			test_scenes = self._load_data_source(data_source, 'test', spatial_boundaries, feature_boundaries)
			print('train')
			train_loader = self._prepare_data(train_scenes, spatial_boundaries, feature_boundaries, self.train_batch_size, False)
			print('test')
			test_loader = self._prepare_data(test_scenes, spatial_boundaries, feature_boundaries, self.test_batch_size, True)
			self.observation_sites[data_source] = EthUcyObservationSite(train_loader, test_loader, spatial_boundaries)
		return self.observation_sites[data_source]
	
	def _prepare_data(self, scenes, spatial_boundaries, feature_boundaries, batch_size, evaluation_set):
		#combined_boundaries = np.concatenate((spatial_boundaries, feature_boundaries), axis=0)
		#for scene in scenes:
		#	for agent in scene.agents:
		#		agent.data = normalize(agent.data, combined_boundaries)

		dataset = EthUcyDataset(scenes, self.history, self.futures, evaluation_set)
		return DataLoader(dataset, batch_size=batch_size, shuffle=True)

	def _load_data_source(self, data_source, data_class, spatial_boundaries, feature_boundaries):
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

					#data['pos_x'] = data['pos_x'] #- data['pos_x'].mean()
					#data['pos_y'] = data['pos_y'] #- data['pos_y'].mean()

					max_timesteps = data['frame_id'].max()

					scene = Scene(max_timesteps)

					for node_id in pd.unique(data['node_id']):
						node_df = data[data['node_id'] == node_id]
						assert np.all(np.diff(node_df['frame_id']) == 1)

						node_values = node_df[['pos_x', 'pos_y']].values

						first_timestep = node_df['frame_id'].iloc[0]

						if node_values.shape[0] < 2:
							continue

						x = node_values[:, 0]
						y = node_values[:, 1]

						dt = 0.4
						vx = derivative_of(x, dt)
						vy = derivative_of(y, dt)
						ax = derivative_of(vx, dt)
						ay = derivative_of(vy, dt)

						spatial_boundaries[0][0] = min(spatial_boundaries[0][0], np.min(x))
						spatial_boundaries[0][1] = max(spatial_boundaries[0][1], np.max(x))
						spatial_boundaries[1][0] = min(spatial_boundaries[1][0], np.min(y))
						spatial_boundaries[1][1] = max(spatial_boundaries[1][1], np.max(y))

						feature_boundaries[0][0] = min(feature_boundaries[0][0], np.min(vx))
						feature_boundaries[0][1] = max(feature_boundaries[0][1], np.max(vx))
						feature_boundaries[1][0] = min(feature_boundaries[1][0], np.min(vy))
						feature_boundaries[1][1] = max(feature_boundaries[1][1], np.max(vy))
						feature_boundaries[2][0] = min(feature_boundaries[2][0], np.min(ax))
						feature_boundaries[2][1] = max(feature_boundaries[2][1], np.max(ax))
						feature_boundaries[3][0] = min(feature_boundaries[3][0], np.min(ay))
						feature_boundaries[3][1] = max(feature_boundaries[3][1], np.max(ay))

						# agents_dir = f"agents_{file.rstrip('.txt')}"
						# os.makedirs(agents_dir, exist_ok=True)
						# filename = os.path.join(agents_dir, f'agent_{len(scene.agents)}.png')
						# if os.path.exists(filename):
						# 	os.remove(filename)
						# plt.figure(figsize=(10, 8))
						# plt.plot(x, y, color='black', linewidth=2, label='Agent Trajectory')
						# plt.scatter(x, y, color='red', s=50, zorder=5)
						# plt.savefig(filename)
						# plt.close()
						#print(f'agent {len(scene.agents)} positions {node_values}')
						#print(x)
						#print(y)

						headers = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc']
						data_dict = {'x_pos': x, 'y_pos': y, 'x_vel': vx, 'y_vel': vy, 'x_acc': ax, 'y_acc': ay}

						node_data = pd.DataFrame(data_dict, columns=headers)
						scene.agents.append(Agent(first_timestep, node_data))
					
					scenes.append(scene)

		return scenes