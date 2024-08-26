import math
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def normalize(data, boundaries):
	return (data - boundaries[:, 0]) / (boundaries[:, 1] - boundaries[:, 0])

def denormalize(data, boundaries):
	return (data * (boundaries[:, 1] - boundaries[:, 0])) + boundaries[:, 0]

class InDDataset(Dataset):
	def __init__(self, input, feature):
		assert input.shape[0] == feature.shape[0]
		self.input = input
		self.feature = feature
		self.data_size = input.size(0)

	def __getitem__(self, index):
		return self.input[index], self.feature[index]

	def __len__(self):
		return self.data_size
	

class InDObservationSite():
	def __init__(self, background, ortho_px_to_meter, boundaries, train_loader, test_loader):
		self.background = background
		self.ortho_px_to_meter = ortho_px_to_meter
		self.boundaries = boundaries
		self.train_loader = train_loader
		self.test_loader = test_loader

	def normalize(self, data):
		return normalize(data, self.boundaries)
	
	def denormalize(self, data):
		return denormalize(data, self.boundaries)
	

class InD():
	def __init__(self, root, train_ratio, train_batch_size, test_batch_size, missing_rate=0):
		self.root = root
		self.train_ratio = train_ratio
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.missing_rate = missing_rate
		self.observation_sites = {}

	@property
	def observation_site1(self):
		#return self._get_observation_site(["07", "08", "09", "10"])
		return self._get_observation_site(["08"])

	def _get_observation_site(self, sites):
		key = '-'.join(sites)
		if key not in self.observation_sites:
			self.observation_sites[key] = self._load_observation_site(sites)
		return self.observation_sites[key]

	def _load_observation_site(self, observation_sites):
		background = os.path.join(f'{self.root}', f'{observation_sites[0]}_background.png')

		spatial_boundaries = np.array([[25, 85], [-65, -10]])
		feature_boundaries = np.array([[0, 360], [-10, 10], [-10, 10], [-5, 5], [-5, 5]])

		ortho_px_to_meter = 0

		train_inputs = []
		test_inputs = []

		train_features = []
		test_features = []

		for observation_site in observation_sites:
			input, features, ortho_px_to_meter = self._parse(observation_site)
			input = normalize(input, spatial_boundaries)
			features = normalize(features, feature_boundaries)

			randidx = np.random.permutation(input.shape[0])
			n_data = len(randidx)
			n_train = int(n_data * self.train_ratio)
			n_test = n_data - n_train

			train_idx = randidx[:n_train]
			test_idx = randidx[n_train:(n_train+n_test)]

			train_input = torch.FloatTensor(input[train_idx])
			test_input = torch.FloatTensor(input[test_idx])

			train_feature = torch.FloatTensor(features[train_idx])
			test_feature = torch.FloatTensor(features[test_idx])

			train_feature = self._append_time(train_feature)
			test_feature = self._append_time(test_feature)

			self._mask(train_input, train_feature)
			self._mask(test_input, test_feature)

			train_inputs.append(train_input)
			test_inputs.append(test_input)

			train_features.append(train_feature)
			test_features.append(test_feature)

		train_input = torch.cat(train_inputs, dim=0)
		train_feature = torch.cat(train_features, dim=0)

		test_input = torch.cat(test_inputs, dim=0)
		test_feature = torch.cat(test_features, dim=0)

		train_data = InDDataset(train_input, train_feature)
		test_data = InDDataset(test_input, test_feature)

		train_loader = DataLoader(dataset=train_data, batch_size=self.train_batch_size, shuffle=True)
		test_loader = DataLoader(dataset=test_data, batch_size=self.test_batch_size, shuffle=True)
		return InDObservationSite(background, ortho_px_to_meter, spatial_boundaries, train_loader, test_loader)
	
	def _parse(self, observation_site):
		recording_metadata = pd.read_csv(os.path.join(f'{self.root}', f'{observation_site}_recordingMeta.csv'))
		tracks = pd.read_csv(os.path.join(f'{self.root}', f'{observation_site}_tracks.csv'))
		tracks_metadata = pd.read_csv(os.path.join(f'{self.root}', f'{observation_site}_tracksMeta.csv'))

		car_list = ['car', 'truck_bus']
		mask = (tracks_metadata['class'] == car_list[0]) | (tracks_metadata['class'] == car_list[1])
		index_df = tracks_metadata[mask]

		numframes = index_df['numFrames']
		target_track_id = index_df[(numframes < 1000) & (numframes > 200)]['trackId']

		new_df = tracks.apply(lambda row: row[tracks['trackId'].isin(target_track_id)])

		moving_window = 200
		raw_data = []
		for id in target_track_id:
			temp_df = new_df[new_df['trackId'] == id]
			assert len(temp_df) == max(temp_df['trackLifetime']) + 1, 'the length should be the same!'
			temp_df_len = len(temp_df)
			for i in range(temp_df_len-moving_window+1):
				temp_df_1 = temp_df[i:i+moving_window]
				assert len(temp_df_1) == moving_window, 'the length should be the moving window!'
				temp_data = temp_df_1.to_numpy()
				raw_data.append(np.expand_dims(temp_data, axis=0))

		data = np.concatenate(raw_data, axis=0)

		input = data[:, :, 4:6]  # x_pos, y_pos
		features = data[:, :, (6, 9, 10, 11, 12)] # heading, x_vel, y_vel, x_acc, y_acc
		ortho_px_to_meter = recording_metadata.at[0, 'orthoPxToMeter']

		return input, features, ortho_px_to_meter
	
	def _append_time(self, features):
		batch_size, seq_len, _ = features.shape
		t = torch.linspace(0., 2., seq_len)
		t = t.unsqueeze(0).unsqueeze(-1)
		t = t.expand(batch_size, seq_len, 1)
		return torch.cat([features, t], dim=-1)
	
	def _mask(self, data, features):
		seq_len = 100 # TODO: we have to not hard code 100 everywhere
		generator = torch.Generator()#.manual_seed(56789)
		for i in range(data.shape[0]):
			mask = torch.randperm(seq_len, generator=generator)[:int(seq_len * self.missing_rate)].sort().values
			data[i][mask] = float('nan')
			features[i][mask] = float('nan')
