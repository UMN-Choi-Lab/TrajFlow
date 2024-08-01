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
	
def construct_causal_enocder(input_dim, hidden_dim, num_layers, causal_encoder):
	if causal_encoder == CausalEnocder.GRU:
		return GRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
	elif causal_encoder == CausalEnocder.LSTM:
		return LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
	elif causal_encoder == CausalEnocder.TRANSFORMER:
		return Transformer(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=4)
	elif causal_encoder == CausalEnocder.CDE:
		return CDE(input_dim=input_dim, embedding_dim=hidden_dim, hidden_dim=128, num_layers=2)
	else:
		raise ValueError(f'{causal_encoder.name} is not a supported causal encoder')


def construct_flow(input_dim, condition_dim, hidden_dims, flow):
	if flow == Flow.DNF:
		return DNF(n_blocks=3, input_size=input_dim, hidden_size=32, n_hidden=10, num_pred=100, cond_label_size=condition_dim)
	elif flow == Flow.CNF:
		return CNF(input_dim, condition_dim, hidden_dims)
	else:
		raise ValueError(f'{flow.name} is not a supported normalizing flow')


class TrajFlow(nn.Module):
	def __init__(self, seq_len, input_dim, feature_dim, embedding_dim, hidden_dims, causal_encoder, flow):
		super(TrajFlow, self).__init__()
		self.causal_encoder = construct_causal_enocder(input_dim + feature_dim, embedding_dim, 3, causal_encoder)
		self.flow = construct_flow(input_dim, embedding_dim, hidden_dims, flow)

		self.register_buffer("base_dist_mean", torch.zeros(seq_len, input_dim))
		self.register_buffer("base_dist_var", torch.ones(seq_len, input_dim))

	@property
	def _base_dist(self):
		return torch.distributions.MultivariateNormal(self.base_dist_mean, torch.diag_embed(self.base_dist_var))

	def _embedding(self, x, feat):
		_, seq_length, _ = x.shape
		x = torch.cat([x, feat], dim=-1)
		t = torch.linspace(0., 1., seq_length).to(x)
		embedding = self.causal_encoder(t, x)
		return embedding

	def forward(self, x, y, feat):
		embedding = self._embedding(x, feat)
		z, delta_logpz = self.flow(y, embedding)
		return z, delta_logpz
	
	def sample(self, x, feat, num_samples=1):
		y = torch.stack([self._base_dist.sample().to(x.device) for _ in range(num_samples)])
		embedding = self._embedding(x, feat)
		#print(y[0, 0, :])
		z, delta_logpz = self.flow(y, embedding, reverse=True)
		#print(z[0, 0, :])
		#y2, _ = self.flow(z, embedding, reverse=False)
		#print(y2[0, 0, :])
		return y, z, delta_logpz

	def log_prob(self, z_t0, delta_logpz):
		logpz_t0 = self._base_dist.log_prob(z_t0)
		logpz_t1 = logpz_t0 - delta_logpz
		return logpz_t0, logpz_t1