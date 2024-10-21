import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
	def __init__(self, scenes, history_frames=8, future_frames=12):
		self.scenes = scenes
		self.history_frames = history_frames
		self.future_frames = future_frames
		self.data = self._prepare_data()

	def _prepare_data(self):
		data = []
		for scene in self.scenes:
			for agent in scene.agents:
				# This idea was good but the foor loop needs to be optimized 
				# if the trajectory is less than 20 its ignored what about 
				# for i in range(0, self.history_frames):
				#   history = agent_data.iloc[max(0, i - self.history_frames), min(len(agent_data) - 1, i + 1)]
				#   future = agent_data.iloc[min(len(agent_data) - 1, i + 1), min(len(agent_data), i + 1 + self.future_frames)]
				for i in range(self.history_frames - 1, len(agent.data) - self.future_frames):
					history = agent.data.iloc[i-self.history_frames+1:i+1]
					future = agent.data.iloc[i+1:i+1+self.future_frames]
					data.append((history, future))
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
		
		# Pad with zeros if necessary... probably can remove
		# if inputs.shape[0] < self.history_frames:
		# 	pad_width = ((self.history_frames - inputs.shape[0], 0), (0, 0))
		# 	inputs = np.pad(inputs, pad_width, mode='constant')

		# if features.shape[0] < self.history_frames:
		# 	pad_width = ((self.history_frames - features.shape[0], 0), (0, 0))
		# 	features = np.pad(features, pad_width, mode='constant')
		
		# if targets.shape[0] < self.future_frames:
		# 	pad_width = ((0, self.future_frames - targets.shape[0]), (0, 0))
		# 	targets = np.pad(targets, pad_width, mode='constant')

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
	def __init__(self, train_batch_size, test_batch_size, history, futures):
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.observation_sites = {}
		self.history = history
		self.futures = futures

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
			# for i in range(25):
			# 	feature_boundaries = np.append(feature_boundaries, [[np.inf, -np.inf]], axis=0)
			train_scenes = self._load_data_source(data_source, 'train', spatial_boundaries, feature_boundaries)
			#train_boundaries, train_loader = self._load_data_source(data_source, 'test', self.train_batch_size)
			test_scenes = self._load_data_source(data_source, 'test', spatial_boundaries, feature_boundaries)
			#test_boundaries, test_loader = self._load_data_source(data_source, 'train', self.test_batch_size)
			train_loader = self._prepare_data(train_scenes, spatial_boundaries, feature_boundaries, self.train_batch_size)
			test_loader = self._prepare_data(test_scenes, spatial_boundaries, feature_boundaries, self.test_batch_size)
			self.observation_sites[data_source] = EthUcyObservationSite(train_loader, test_loader, spatial_boundaries)
		return self.observation_sites[data_source]
	
	def _prepare_data(self, scenes, spatial_boundaries, feature_boundaries, batch_size):
		combined_boundaries = np.concatenate((spatial_boundaries, feature_boundaries), axis=0)
		for scene in scenes:
			for agent in scene.agents:
				agent.data = normalize(agent.data, combined_boundaries)

		dataset = EthUcyDataset(scenes, self.history, self.futures)
		return DataLoader(dataset, batch_size=batch_size, shuffle=True)

	def _load_data_source(self, data_source, data_class, spatial_boundaries, feature_boundaries):
		N = 5 # neighboorhood size
		scenes = []
		first = True # remove me
		# for i in range(N * N):
		# 	feature_boundaries = np.append(feature_boundaries, [[np.inf, -np.inf]], axis=0)
		
		for subdir, _, files in os.walk(os.path.join('data', 'raw', data_source, data_class)):
			for file in files:
				if first and file.endswith('.txt'):
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

					# This is slow we need to optimize
					
					# grid_size = 10
					# cell_size = grid_size / N
					# half_grid = grid_size // 2

					# for frame_id in range(max_timesteps):
					# 	agents = data[data['frame_id'] == frame_id]
					# 	agent_ids = agents['node_id']

					# 	for agent_id in agent_ids:
					# 		agent_data = agents[agents['node_id'] == agent_id]
					# 		x_pos = agent_data['pos_x'].values[0]
					# 		y_pos = agent_data['pos_y'].values[0]

					# 		counts = np.zeros((N, N))
					# 		for i in range(N):
					# 			for j in range(N):
					# 				x_min = x_pos - half_grid + i * cell_size
					# 				x_max = x_pos - half_grid + (i + 1) * cell_size
					# 				y_min = y_pos - half_grid + j * cell_size
					# 				y_max = y_pos - half_grid + (j + 1) * cell_size
					# 				counts[i, j] = agents[(agents['node_id'] != agent_id) &
					# 						(agents['pos_x'] >= x_min) & (agents['pos_x'] < x_max) &
					# 						(agents['pos_y'] >= y_min) & (agents['pos_y'] < y_max)].shape[0]

					# 		social_occupancy = counts.flatten().tolist()
					# 		mask = (data['frame_id'] == frame_id) & (data['node_id'] == agent_id)

					# 		for i in range(len(social_occupancy)):
					# 			data.loc[mask, f'social_occupancy{i}'] = social_occupancy[i]

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

						# agents_dir = "agents"
						# os.makedirs(agents_dir, exist_ok=True)
						# filename = os.path.join(agents_dir, f'agent_{len(scene.agents)}.png')
						# if os.path.exists(filename):
						# 	os.remove(filename)
						# plt.figure(figsize=(10, 8))
						# plt.plot(x, y, color='black', linewidth=2, label='Agent Trajectory')
						# plt.scatter(x, y, color='red', s=50, zorder=5)
						# plt.savefig(filename)
						# plt.close()
						# print(f'agent {len(scene.agents)} positions {node_values}')
						# print(x)
						# print(y)

						headers = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc']
						data_dict = {'x_pos': x, 'y_pos': y, 'x_vel': vx, 'y_vel': vy, 'x_acc': ax, 'y_acc': ay}

						# for i in range(N * N):
						# 	social_occupancy = node_df[f'social_occupancy{i}'].values
						# 	data_dict[f'social_occupancy{i}'] = social_occupancy
						# 	headers.append(f'social_occupancy{i}')
						# 	feature_boundaries[4 + i][0] = min(feature_boundaries[3 + i][0], np.min(social_occupancy))
						# 	feature_boundaries[4 + i][1] = max(feature_boundaries[3 + i][1], np.max(social_occupancy))

						node_data = pd.DataFrame(data_dict, columns=headers)
						scene.agents.append(Agent(first_timestep, node_data))
					
					scenes.append(scene)

		return scenes