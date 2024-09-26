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

# TODO: We still need to normalize...

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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
				agent_data = agent.data
				# This idea was good but the foor loop needs to be optimized 
				# if the trajectory is less than 20 its ignored what about 
				# for i in range(0, self.history_frames):
				#   history = agent_data.iloc[max(0, i - self.history_frames), min(len(agent_data) - 1, i + 1)]
				#   future = agent_data.iloc[min(len(agent_data) - 1, i + 1), min(len(agent_data), i + 1 + self.future_frames)]
				for i in range(self.history_frames - 1, len(agent_data) - self.future_frames):
					history = agent_data.iloc[i-self.history_frames+1:i+1]
					future = agent_data.iloc[i+1:i+1+self.future_frames]
					data.append((history, future))
		return data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		history, future = self.data[idx]

		input_columns = ['x_pos', 'y_pos']
		feature_columns = [col for col in history.columns if col not in input_columns]

		inputs = history[input_columns].values
		features = history[feature_columns].values
		targets = future[input_columns].values
		
		# Pad with zeros if necessary
		if inputs.shape[0] < self.history_frames:
			pad_width = ((self.history_frames - inputs.shape[0], 0), (0, 0))
			inputs = np.pad(inputs, pad_width, mode='constant')

		if features.shape[0] < self.history_frames:
			pad_width = ((self.history_frames - features.shape[0], 0), (0, 0))
			features = np.pad(features, pad_width, mode='constant')
		
		if targets.shape[0] < self.future_frames:
			pad_width = ((0, self.future_frames - targets.shape[0]), (0, 0))
			targets = np.pad(targets, pad_width, mode='constant')
		
		return inputs, features, targets
	
class EthUcyObservationSite():
	def __init__(self, train_loader, test_loader):
		self.train_loader = train_loader
		self.test_loader = test_loader

class EthUcy():
	def __init__(self):
		self.scenes = []
		self.dt = 0.4

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
		train_loader = self._load_data_source(data_source, 'train')
		test_loader = self._load_data_source(data_source, 'test')
		observation_site = EthUcyObservationSite(train_loader, test_loader)
		return observation_site

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

					max_timesteps = data['frame_id'].max()

					# This is slow we need to optimize
					N = 5
					grid_size = 10
					cell_size = grid_size / N
					half_grid = grid_size // 2

					for frame_id in range(max_timesteps):
						agents = data[data['frame_id'] == frame_id]
						agent_ids = agents['node_id']

						for agent_id in agent_ids:
							agent_data = agents[agents['node_id'] == agent_id]
							x_pos = agent_data['pos_x'].values[0]
							y_pos = agent_data['pos_y'].values[0]

							counts = np.zeros((N, N))
							for i in range(N):
								for j in range(N):
									x_min = x_pos - half_grid + i * cell_size
									x_max = x_pos - half_grid + (i + 1) * cell_size
									y_min = y_pos - half_grid + j * cell_size
									y_max = y_pos - half_grid + (j + 1) * cell_size
									counts[i, j] = agents[(agents['node_id'] != agent_id) &
											(agents['pos_x'] >= x_min) & (agents['pos_x'] < x_max) &
											(agents['pos_y'] >= y_min) & (agents['pos_y'] < y_max)].shape[0]

							social_occupancy = counts.flatten().tolist()
							mask = (data['frame_id'] == frame_id) & (data['node_id'] == agent_id)

							for i in range(len(social_occupancy)):
								data.loc[mask, f'social_occupancy{i}'] = social_occupancy[i]

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
						vx = derivative_of(x, self.dt)
						vy = derivative_of(y, self.dt)
						ax = derivative_of(vx, self.dt)
						ay = derivative_of(vy, self.dt)

						headers = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc']
						data_dict = {'x_pos': x, 'y_pos': y, 'x_vel': vx, 'y_vel': vy, 'x_acc': ax, 'y_acc': ay}

						for i in range(N * N):
							data_dict[f'social_occupancy{i}'] = node_df[f'social_occupancy{i}'].values
							headers.append(f'social_occupancy{i}')

						node_data = pd.DataFrame(data_dict, columns=headers)
						scene.agents.append(Agent(first_timestep, node_data))
					
					scenes.append(scene)

		dataset = EthUcyDataset(scenes)
		return DataLoader(dataset, batch_size=32, shuffle=True)


# dataset = EthUcy()
# first = True
# for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
# 	for data_class in ['train', 'val', 'test']:
# 		if first:
# 			print(f"Loading {desired_source} {data_class}")
# 			scenes = dataset._load_data_source(desired_source, data_class)
# 			dataset.scenes.extend(scenes)
# 			first = False

# data_loader = dataset.create_data_loader(batch_size=64)
dataset = EthUcy()
observation_site = dataset.eth_observation_site
data_loader = observation_site.train_loader

print("Data loader created. Starting iteration...")
for i, (batch_input, batch_feature, batch_target) in enumerate(data_loader):
	print(f"Batch {i+1}:")
	print(f"Input shape (batch_input): {batch_input.shape}")
	print(f"Feature shape (batch_feature): {batch_feature.shape}")
	print(f"Target shape (batch_target): {batch_target.shape}")

print("Data loading test completed.")