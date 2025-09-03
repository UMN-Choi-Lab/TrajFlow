import torch
import torch.nn as nn
from model.encoder.GRU import GRU
from model.encoder.LSTM import LSTM
from model.encoder.Transformer import Transformer
from model.encoder.CDE import CDE
from model.flow.CNF import CNF
from model.flow.DNF import DNF

from enum import Enum

class CausalEnocder(Enum):
	GRU = 1
	LSTM = 2
	TRANSFORMER = 3
	CDE = 4
	
class Flow(Enum):
	DNF = 1
	CNF = 2
	
def construct_causal_enocder(input_dim, embedding_dim, hidden_dim, num_layers, causal_encoder):
	if causal_encoder == CausalEnocder.GRU:
		return GRU(input_dim=input_dim, hidden_dim=embedding_dim, num_layers=num_layers)
	elif causal_encoder == CausalEnocder.LSTM:
		return LSTM(input_dim=input_dim, hidden_dim=embedding_dim, num_layers=num_layers)
	elif causal_encoder == CausalEnocder.TRANSFORMER:
		return Transformer(input_dim=input_dim, hidden_dim=embedding_dim, num_layers=num_layers, num_heads=num_layers)
	elif causal_encoder == CausalEnocder.CDE:
		return CDE(input_dim=input_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)
	else:
		raise ValueError(f'{causal_encoder.name} is not a supported causal encoder')


def construct_flow(input_dim, condition_dim, hidden_dim, flow, marginal):
	if flow == Flow.DNF:
		return DNF(n_blocks=3, input_size=input_dim, hidden_size=hidden_dim, n_hidden=10, 
			cond_label_size=condition_dim, marginal=marginal)
	elif flow == Flow.CNF:
		return CNF(input_dim, condition_dim, (hidden_dim for _ in range(4)), marginal)
	else:
		raise ValueError(f'{flow.name} is not a supported normalizing flow')


class TrajFlow(nn.Module):
	def __init__(self, 
			  seq_len, input_dim, feature_dim, embedding_dim, hidden_dim, 
			  causal_encoder, flow, marginal=False, norm_rotation=True):
		super(TrajFlow, self).__init__()
		self.seq_len = seq_len
		self.input_dim = input_dim
		self.feature_dim = feature_dim
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		self.marginal = marginal
		self.norm_rotation = norm_rotation
		flow_input_dim = input_dim if marginal else seq_len * input_dim

		self.causal_encoder = construct_causal_enocder(input_dim + feature_dim, embedding_dim, hidden_dim, 4, causal_encoder)
		self.flow = construct_flow(flow_input_dim, embedding_dim, hidden_dim, flow, marginal)
	
	def _abs_to_rel(self, y, x_t):
		y_rel = y - x_t
		y_rel[:,1:] = (y_rel[:,1:] - y_rel[:,:-1])
		return y_rel

	def _rel_to_abs(self, y_rel, x_t):
		y_abs = torch.cumsum(y_rel, dim=-2) + x_t 
		return y_abs
	
	def _rotate(self, x, x_t, angles_rad):
		c, s = torch.cos(angles_rad), torch.sin(angles_rad)
		c, s = c.unsqueeze(1), s.unsqueeze(1)
		x_center = x - x_t
		x_vals, y_vals = x_center[:, :, 0], x_center[:, :, 1]
		new_x_vals = c * x_vals + (-1 * s) * y_vals
		new_y_vals = s * x_vals + c * y_vals
		x_center[:, :, 0] = new_x_vals
		x_center[:, :, 1] = new_y_vals
		return x_center + x_t
	
	def _rotate_features(self, features, angles_rad):
		c, s = torch.cos(angles_rad), torch.sin(angles_rad)
		c, s = c.unsqueeze(1), s.unsqueeze(1)
		vx_vals, vy_vals = features[:, :, 0], features[:, :, 1]
		ax_vals, ay_vals = features[:, :, 2], features[:, :, 3]
		# what about headings?
		new_vx_vals = c * vx_vals + (-1 * s) * vy_vals
		new_vy_vals = s * vx_vals + c * vy_vals
		new_ax_vals = c * ax_vals + (-1 * s) * ay_vals
		new_ay_vals = s * ax_vals + c * ay_vals
		features[:, :, 0] = new_vx_vals
		features[:, :, 1] = new_vy_vals
		features[:, :, 2] = new_ax_vals
		features[:, :, 3] = new_ay_vals
		return features

	def _normalize_rotation(self, x, y_true=None):
		x_t = x[:, -1:, :]
		x_t_rel = x[:, -1] - x[:, -2]
		rot_angles_rad = -1 * torch.atan2(x_t_rel[:, 1], x_t_rel[:, 0])
		x = self._rotate(x, x_t, rot_angles_rad)

		if y_true != None:
			y_true = self._rotate(y_true, x_t, rot_angles_rad)
			return x, y_true, rot_angles_rad

		return x, rot_angles_rad
	
	def _embedding(self, x, feat): 
		_, seq_length, _ = x.shape
		x = torch.cat([x, feat], dim=-1)
		t = torch.linspace(0., 2., 2 * seq_length).to(x)
		t = t[:seq_length]
		embedding = self.causal_encoder(t, x)
		return embedding

	def forward(self, x, y, feat, sampling_frequency=1):
		if self.norm_rotation:
			x, y, angle = self._normalize_rotation(x, y)
			feat = self._rotate_features(feat, angle)

		if not self.marginal:
			x_t = x[...,-1:,:]
			y = self._abs_to_rel(y, x_t)

		batch, seq_len, input_dim = y.shape
		y = y if self.marginal else y.view(batch, seq_len * input_dim)
		
		embedding = self._embedding(x, feat)
		z, delta_logpz = self.flow(y, embedding, sampling_frequency=sampling_frequency)
		
		z = z if self.marginal else z.view(batch, seq_len, input_dim)
		return z, delta_logpz
	
	def sample(self, x, feat, futures, num_samples=1, sampling_frequency=1):
		if self.norm_rotation:
			x, angle = self._normalize_rotation(x)
			feat = self._rotate_features(feat, angle)

		mean = (torch.zeros(futures, self.input_dim) if self.marginal else torch.zeros(self.seq_len * self.input_dim)).to(x.device)
		variance = (torch.ones(futures, self.input_dim) if self.marginal else torch.ones(self.seq_len * self.input_dim)).to(x.device)
		base_dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(variance))

		y = torch.stack([base_dist.sample().to(x.device) for _ in range(num_samples)])
		embedding = self._embedding(x, feat)
		embedding = embedding.expand(y.shape[0], embedding.shape[1])
		z, delta_logpz = self.flow(y, embedding, reverse=True, sampling_frequency=sampling_frequency)

		if not self.marginal:
			output_shape = (x.size(0), num_samples, self.seq_len, 2)
			z = z.view(*output_shape)
			x_t = x[..., -1:, :].unsqueeze(dim=1).repeat(1, num_samples, 1, 1)
			z = self._rel_to_abs(z, x_t)[0]

		if self.norm_rotation:
			x_t = x[..., -1:, :]
			z = self._rotate(z, x_t, -1 * angle)
		
		y = y if self.marginal else y.view(y.shape[0], self.seq_len, self.input_dim)
		z = z[:, :futures, :]
		return y, z, delta_logpz

	def log_prob(self, z_t0, delta_logpz):
		batch, seq_len, input_dim = z_t0.shape
		z_t0 = z_t0 if self.marginal else z_t0.view(batch, seq_len * input_dim)

		mean = (torch.zeros(seq_len, input_dim) if self.marginal else torch.zeros(seq_len * input_dim)).to(z_t0.device)
		variance = (torch.ones(seq_len, input_dim) if self.marginal else torch.ones(seq_len * input_dim)).to(z_t0.device)
		base_dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(variance))
		logpz_t0 = base_dist.log_prob(z_t0)

		logpz_t1 = logpz_t0 - delta_logpz
		return logpz_t0, logpz_t1