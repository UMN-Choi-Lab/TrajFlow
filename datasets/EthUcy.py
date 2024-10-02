# import os
# import numpy as np
# import pandas as pd

# def make_continuous_copy(alpha):
#     alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
#     continuous_x = np.zeros_like(alpha)
#     continuous_x[0] = alpha[0]
#     for i in range(1, len(alpha)):
#         if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
#             continuous_x[i] = continuous_x[i - 1] + (
#                     alpha[i] - alpha[i - 1]) - np.sign(
#                 (alpha[i] - alpha[i - 1])) * 2 * np.pi
#         else:
#             continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

#     return continuous_x


# def derivative_of(x, dt=1, radian=False):
#     if radian:
#         x = make_continuous_copy(x)

#     not_nan_mask = ~np.isnan(x)
#     masked_x = x[not_nan_mask]

#     if masked_x.shape[-1] < 2:
#         return np.zeros_like(x)

#     dx = np.full_like(x, np.nan)
#     dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

#     return dx

# class Scene():
#     def __init__(self, timesteps):
#         self.timesteps = timesteps
#         self.agents = []

# class Agent():
#     def __init__(self, first_timestep, data):
#         self.first_timestep = first_timestep
#         self.data = data

# class EthUcy():
#     def __init__(self):
#         self.scenes = []
#         self.dt = 0.4
#         #self.data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
#         # headers = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc']
#         # for i in range(25):
#         #     headers.append(f'social_occupancy{i}')
#         # self.data_columns = pd.Index.from_product(headers)

#     def _load_data_soure(self, data_source, data_class):
#         scenes = []
		
#         for subdir, _, files in os.walk(os.path.join('data', 'raw', data_source, data_class)):
#             for file in files:
#                 if file.endswith('.txt'):
#                     full_data_path = os.path.join(subdir, file)
#                     print('At', full_data_path)

#                     data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
#                     data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
#                     data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
#                     data['frame_id'] = data['frame_id'] // 10
#                     data['frame_id'] -= data['frame_id'].min()
#                     data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

#                     data['node_id'] = data['track_id'].astype(str)
#                     data.sort_values('frame_id', inplace=True)

#                     # Mean Position
#                     data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
#                     data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

#                     max_timesteps = data['frame_id'].max()

#                     N = 5
#                     grid_size = 10
#                     cell_size = grid_size / N
#                     half_grid = grid_size // 2

#                     for frame_id in range(max_timesteps):
#                         agents = data[data['frame_id'] == frame_id]
#                         agent_ids = agents['node_id']

#                         for agent_id in agent_ids:
#                             agent_data = agents[agents['node_id'] == agent_id]
#                             x_pos = agent_data['pos_x'].values[0]
#                             y_pos = agent_data['pos_y'].values[0]

#                             counts = np.zeros((N, N))
#                             for i in range(N):
#                                 for j in range(N):
#                                     x_min = x_pos - half_grid + i * cell_size
#                                     x_max = x_pos - half_grid + (i + 1) * cell_size
#                                     y_min = y_pos - half_grid + j * cell_size
#                                     y_max = y_pos - half_grid + (j + 1) * cell_size
#                                     counts[i, j] = agents[(agents['node_id'] != agent_id) &
#                                             (agents['pos_x'] >= x_min) & (agents['pos_x'] < x_max) &
#                                             (agents['pos_y'] >= y_min) & (agents['pos_y'] < y_max)].shape[0]

#                             social_occupancy = counts.flatten().tolist()
#                             mask = (data['frame_id'] == frame_id) & (data['node_id'] == agent_id)

#                             for i in range(len(social_occupancy)):
#                                 data.loc[mask, f'social_occupancy{i}'] = social_occupancy[i]

#                     scene = Scene(max_timesteps)

#                     for node_id in pd.unique(data['node_id']):
#                         node_df = data[data['node_id'] == node_id]
#                         assert np.all(np.diff(node_df['frame_id']) == 1)

#                         node_values = node_df[['pos_x', 'pos_y']].values

#                         first_timestep = node_df['frame_id'].iloc[0]

#                         if node_values.shape[0] < 2:
#                             continue

#                         x = node_values[:, 0]
#                         y = node_values[:, 1]
#                         vx = derivative_of(x, self.dt)
#                         vy = derivative_of(y, self.dt)
#                         ax = derivative_of(vx, self.dt)
#                         ay = derivative_of(vy, self.dt)

#                         headers = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc']
#                         data_dict = {'x_pos': x, 'y_pos': y, 'x_vel': vx, 'y_vel': vy, 'x_acc': ax, 'y_acc': ay}

#                         for i in range(N * N):
#                             data_dict[f'social_occupancy{i}'] = node_df[f'social_occupancy{i}'].values
#                             headers.append(f'social_occupancy{i}')

#                         node_data = pd.DataFrame(data_dict, columns=headers)
#                         scene.agents.append(Agent(first_timestep, node_data))
					
#                     scenes.append(scene)

#         return scenes

# # For each agent at each timestep
# # x is the previous 8 frames for that agent (including this one)
# # y is the next 12 frames for that agent
# # in the case of missing data we can pad with zeros or something

# dataset = EthUcy()
# first = True
# for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
#     for data_class in ['train', 'val', 'test']:
#         if first:
#             print(f'{desired_source} {data_class}')
#             scenes = dataset._load_data_soure(desired_source, data_class)
#             for scene in scenes:
#                 print(scene.timestep)
#                 print(len(scene.agents))
#                 for agent in scene.agents:
#                     print(agent.first_timestep)
#             first = False

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

def make_continuous_copy(alpha):
	alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
	continuous_x = np.zeros_like(alpha)
	continuous_x[0] = alpha[0]
	for i in range(1, len(alpha)):
		if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
			continuous_x[i] = continuous_x[i - 1] + (
					alpha[i] - alpha[i - 1]) - np.sign(
				(alpha[i] - alpha[i - 1])) * 2 * np.pi
		else:
			continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

	return continuous_x

def derivative_of(x, dt=1, radian=False):
	if radian:
		x = make_continuous_copy(x)

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
	def __init__(self, train_batch_size, test_batch_size):
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.observation_sites = {}

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

		dataset = EthUcyDataset(scenes)
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


# dataset = EthUcy(1, 1)
# test = dataset.eth_observation_site.test_loader
# sample = next(iter(test))

# # dataset = EthUcy()
# # first = True
# # for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
# # 	for data_class in ['train', 'val', 'test']:
# # 		if first:
# # 			print(f"Loading {desired_source} {data_class}")
# # 			scenes = dataset._load_data_source(desired_source, data_class)
# # 			dataset.scenes.extend(scenes)
# # 			first = False

# # data_loader = dataset.create_data_loader(batch_size=64)
# dataset = EthUcy()
# observation_site = dataset.eth_observation_site
# data_loader = observation_site.train_loader

# print("Data loader created. Starting iteration...")
# for i, (batch_input, batch_feature, batch_target) in enumerate(data_loader):
# 	print(f"Batch {i+1}:")
# 	print(f"Input shape (batch_input): {batch_input.shape}")
# 	print(f"Feature shape (batch_feature): {batch_feature.shape}")
# 	print(f"Target shape (batch_target): {batch_target.shape}")

# print("Data loading test completed.")

# import os
# import pickle
# import torch
# from torch.utils.data import Dataset, DataLoader

# class EthUcyDataset(Dataset):
# 	def __init__(self, filepath):
# 		self.inputs, self.features, self.targets = self._prepare_data(filepath)

# 	def _prepare_data(self, filepath):
# 		with open(filepath, 'rb') as file:
# 			restored_batches = pickle.load(file)
	
# 		x_t_stack = []
# 		y_t_stack = []
# 		x_st_t_stack = []
# 		y_st_t_stack = []

# 		for batch in restored_batches:
# 			x_t = batch['x_t']
# 			y_t = batch['y_t']
# 			x_st_t = batch['x_st_t']
# 			y_st_t = batch['y_st_t']

# 			x_t_stack.append(x_t)
# 			y_t_stack.append(y_t)
# 			x_st_t_stack.append(x_st_t)
# 			y_st_t_stack.append(y_st_t)

# 		x = torch.cat(x_t_stack, dim=0)
# 		y = torch.cat(y_t_stack, dim=0)
# 		x_st = torch.cat(x_st_t_stack, dim=0)
# 		y_st = torch.cat(y_st_t_stack, dim=0)

# 		# remove me just for the sake of GRU
# 		x_st = torch.nan_to_num(x_st, nan=0.0)

# 		inputs = x_st[:, :, :2] 
# 		features = x_st[:, :, 2:]
# 		features = self._append_time(features)
# 		targets = y_st

# 		return inputs, features, targets
	
# 	def _append_time(self, features):
# 		batch_size, seq_len, _ = features.shape
# 		t = torch.linspace(0., 1., seq_len)
# 		t = t.unsqueeze(0).unsqueeze(-1)
# 		t = t.expand(batch_size, seq_len, 1)
# 		return torch.cat([features, t], dim=-1)

# 	def __len__(self):
# 		return self.inputs.shape[0]

# 	def __getitem__(self, idx):
# 		return self.inputs[idx], self.features[idx], self.targets[idx]
	
# class EthUcyObservationSite():
# 	def __init__(self, train_loader, test_loader):
# 		self.train_loader = train_loader
# 		self.test_loader = test_loader
	
# class EthUcy():
# 	def __init__(self, train_batch_size, test_batch_size):
# 		self.train_batch_size = train_batch_size
# 		self.test_batch_size = test_batch_size
# 		self.observation_sites = {}

# 	@property
# 	def eth_observation_site(self):
# 		return self._get_observation_site('eth')
	
# 	@property
# 	def hotel_observation_site(self):
# 		return self._get_observation_site('hotel')
	
# 	@property
# 	def univ_observation_site(self):
# 		return self._get_observation_site('univ')
	
# 	@property
# 	def zara1_observation_site(self):
# 		return self._get_observation_site('zara1')
	
# 	@property
# 	def zara2_observation_site(self):
# 		return self._get_observation_site('zara2')
	
# 	def _get_observation_site(self, data_source):
# 		if data_source not in self.observation_sites:
# 			train_loader = self._load_data_source(data_source, 'train', self.train_batch_size)
# 			test_loader = self._load_data_source(data_source, 'test', self.test_batch_size)
# 			self.observation_sites[data_source] = EthUcyObservationSite(train_loader, test_loader)
# 		return self.observation_sites[data_source]

# 	def _load_data_source(self, data_source, data_class, batch_size):
# 		filepath = os.path.join('data', f'{data_source}_{data_class}.pkl')
# 		dataset = EthUcyDataset(filepath)
# 		return DataLoader(dataset, batch_size=batch_size, shuffle=True)
	

# ethucy = EthUcy(train_batch_size=128, test_batch_size=1)
# train = ethucy.eth_observation_site.train_loader
# inputs, features, targets = next(iter(train))
# print(inputs.shape)
# print(features.shape)
# print(targets.shape)

# print(inputs[0])
# print(features[0])
# print(targets[0])

# filename = 'eth_test.pkl'

# with open(filename, 'rb') as file:
# 	restored_batches = pickle.load(file)
	
# x_t_stack = []
# y_t_stack = []
# x_st_t_stack = []
# y_st_t_stack = []

# for batch in restored_batches:
# 	x_t = batch['x_t']
# 	y_t = batch['y_t']
# 	x_st_t = batch['x_st_t']
# 	y_st_t = batch['y_st_t']

# 	x_t_stack.append(x_t)
# 	y_t_stack.append(y_t)
# 	x_st_t_stack.append(x_st_t)
# 	y_st_t_stack.append(y_st_t)

# x = torch.cat(x_t_stack, dim=0)
# y = torch.cat(y_t_stack, dim=0)
# x_st = torch.cat(x_st_t_stack, dim=0)
# y_st = torch.cat(y_st_t_stack, dim=0)

# print("All Samples Tensors Shapes:")
# print(x.shape)
# print(y.shape)
# print(x_st.shape)
# print(y_st.shape)

# #we can zero pad I guess..., do this for GRU
# #x = torch.nan_to_num(x, nan=0.0)

# print(torch.any(torch.isnan(x)))
# print(torch.any(torch.isnan(y)))