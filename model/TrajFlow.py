import math
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
	
def construct_causal_enocder(input_dim, embedding_dim, num_layers, causal_encoder):
	if causal_encoder == CausalEnocder.GRU:
		return GRU(input_dim=input_dim, hidden_dim=embedding_dim, num_layers=num_layers)
	elif causal_encoder == CausalEnocder.LSTM:
		return LSTM(input_dim=input_dim, hidden_dim=embedding_dim, num_layers=num_layers)
	elif causal_encoder == CausalEnocder.TRANSFORMER:
		return Transformer(input_dim=input_dim, hidden_dim=embedding_dim, num_layers=num_layers, num_heads=4)
	elif causal_encoder == CausalEnocder.CDE:
		return CDE(input_dim=input_dim, embedding_dim=embedding_dim, hidden_dim=512, num_layers=num_layers)
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
	def __init__(self, seq_len, input_dim, feature_dim, embedding_dim, hidden_dim, causal_encoder, flow, marginal=False):
		super(TrajFlow, self).__init__()
		self.marginal = marginal
		self.seq_len = seq_len
		self.input_dim = input_dim
		self.alpha = 10

		flow_input_dim = input_dim if marginal else seq_len * input_dim

		self.causal_encoder = construct_causal_enocder(input_dim + feature_dim, embedding_dim, 4, causal_encoder)
		self.flow = construct_flow(flow_input_dim, embedding_dim, hidden_dim, flow, marginal)

		if marginal:
			self.register_buffer("base_dist_mean", torch.zeros(seq_len, input_dim))
			self.register_buffer("base_dist_var", torch.ones(seq_len, input_dim))
		else:
			self.register_buffer("base_dist_mean", torch.zeros(seq_len * input_dim))
			self.register_buffer("base_dist_var", torch.ones(seq_len * input_dim))

	@property
	def _base_dist(self):
		return torch.distributions.MultivariateNormal(self.base_dist_mean, torch.diag_embed(self.base_dist_var))
	
	def _abs_to_rel(self, y, x_t):
		y_rel = y - x_t
		y_rel[:,1:] = (y_rel[:,1:] - y_rel[:,:-1])
		y_rel = y_rel * self.alpha
		return y_rel

	def _rel_to_abs(self, y_rel, x_t):
		y_abs = y_rel / self.alpha
		return torch.cumsum(y_abs, dim=-2) + x_t 

	def _embedding(self, x, feat):
		_, seq_length, _ = x.shape
		x = torch.cat([x, feat], dim=-1)
		t = torch.linspace(0., 2., 2 * seq_length).to(x)
		t = t[:seq_length]
		embedding = self.causal_encoder(t, x)
		return embedding

	def forward(self, x, y, feat):
		if not self.marginal:
			x_t = x[...,-1:,:]
			y = self._abs_to_rel(y, x_t)
		batch, seq_len, input_dim = y.shape
		y = y if self.marginal else y.view(batch, seq_len * input_dim)
		embedding = self._embedding(x, feat)
		z, delta_logpz = self.flow(y, embedding)
		z = z if self.marginal else z.view(batch, seq_len, input_dim)
		return z, delta_logpz
	
	def sample(self, x, feat, num_samples=1):
		y = torch.stack([self._base_dist.sample().to(x.device) for _ in range(num_samples)])
		embedding = self._embedding(x, feat)
		embedding = embedding.expand(y.shape[0], embedding.shape[1])
		z, delta_logpz = self.flow(y, embedding, reverse=True)
		if not self.marginal:
			output_shape = (x.size(0), num_samples, self.seq_len, 2)
			z = z.view(*output_shape)
			x_t = x[...,-1:,:]
			#x_t = x_t[0].repeat(1, self.seq_len).unsqueeze(0)
			x_t = x[..., -1:, :].unsqueeze(dim=1).repeat(1, num_samples, 1, 1)
			z = self._rel_to_abs(z, x_t)[0]
			#z = self._rel_to_abs(z, x_t)[0]
		#z = z if self.marginal else z.view(z.shape[0], self.seq_len, self.input_dim)
		return y, z, delta_logpz # y might not be the correct shape for joint densities

	def log_prob(self, z_t0, delta_logpz):
		batch, seq_len, input_dim = z_t0.shape
		z_t0 = z_t0 if self.marginal else z_t0.view(batch, seq_len * input_dim)
		#logpz_t0 = self._base_dist.log_prob(z_t0)
		#begin experimental
		mean = torch.zeros(seq_len, input_dim).to(z_t0.device)
		variance = torch.ones(seq_len, input_dim).to(z_t0.device)
		base_dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(variance))
		logpz_t0 = base_dist.log_prob(z_t0)
		#end experimental
		logpz_t1 = logpz_t0 - delta_logpz
		return logpz_t0, logpz_t1