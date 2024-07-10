import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint

class ConditionalODE(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims):
		super(ConditionalODE, self).__init__()
		dim_list = [input_dim + condition_dim] + list(hidden_dims) + [input_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i] + 2, dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.LayerNorm(dim_list[i + 1]))
		self.layers = nn.ModuleList(layers)
		self.condition = None

	def _z_dot(self, t, z):
		positional_encoding = (torch.cumsum(torch.ones_like(z)[:, :, 0], 1) / z.shape[1]).unsqueeze(-1)
		time_encoding = t.expand(z.shape[0], z.shape[1], 1)
		condition = self.condition.unsqueeze(1).expand(-1, z.shape[1], -1)
		z_dot = torch.cat([z, condition], dim=-1)
		for i in range(0, len(self.layers), 2):
			tpz_cat = torch.cat([time_encoding, positional_encoding, z_dot], dim=-1)
			z_dot = self.layers[i](tpz_cat)
			if i < len(self.layers) - 2:
				z_dot = self.layers[i + 1](z_dot)
				z_dot = F.softplus(z_dot)
		return z_dot
	
	def _jacobian_trace(seld, z_dot, z):
		batch_size, seq_len, dim = z.shape
		trace = torch.zeros(batch_size, seq_len, device=z.device)
		for i in range(dim):
			trace += torch.autograd.grad(z_dot[:, :, i].sum(), z, create_graph=True)[0][:, :, i]
		return trace
	
	def forward(self, t, states):
		z = states[0]

		with torch.set_grad_enabled(True):
			z.requires_grad_(True)
			t.requires_grad_(True)
			z_dot = self._z_dot(t, z)
			divergence = self._jacobian_trace(z_dot, z)

		return z_dot, -divergence


class ConditionalCNF(torch.nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims):
		super(ConditionalCNF, self).__init__()
		self.time_derivative = ConditionalODE(input_dim, condition_dim, hidden_dims)

	def forward(self, z, condition, delta_logpz=None, integration_times=None, reverse=False):
		if delta_logpz is None:
			delta_logpz = torch.zeros(z.shape[0], z.shape[1]).to(z)
		if integration_times is None:
			integration_times = torch.tensor([0.0, 1.0]).to(z)
		if reverse:
			integration_times = torch.flip(integration_times, [0])

		self.time_derivative.condition = condition
		state = odeint_adjoint(self.time_derivative, (z, delta_logpz), integration_times, method='dopri5', atol=1e-5, rtol=1e-5)

		if len(integration_times) == 2:
			state = tuple(s[1] for s in state)
		z, delta_logpz = state
		return z, delta_logpz
	

class TrajCNF_GRU(torch.nn.Module):
	def __init__(self, seq_len, input_dim, feature_dim, embedding_dim, hidden_dims):
		super(TrajCNF_GRU, self).__init__()
		self.causal_encoder = nn.GRU(input_dim + feature_dim, embedding_dim, num_layers=3, batch_first=True)
		self.flow = ConditionalCNF(input_dim, embedding_dim, hidden_dims)

		self.register_buffer("base_dist_mean", torch.zeros(seq_len, input_dim))
		self.register_buffer("base_dist_var", torch.ones(seq_len, input_dim))

	@property
	def _base_dist(self):
		return torch.distributions.MultivariateNormal(self.base_dist_mean, torch.diag_embed(self.base_dist_var))

	def _embedding(self, x, feat):
		x = torch.cat([x, feat], dim=-1)
		embedding, _ = self.causal_encoder(x)
		return embedding[:, -1, :]

	def forward(self, x, y, feat):
		embedding = self._embedding(x, feat)
		z, delta_logpz = self.flow(y, embedding)
		return z, delta_logpz
	
	def sample(self, x, feat, num_samples=1):
		y = torch.stack([self._base_dist.sample().to(x.device) for _ in range(num_samples)])
		embedding = self._embedding(x, feat)
		z, delta_logpz = self.flow(y, embedding, reverse=True)
		return y, z, delta_logpz

	def log_prob(self, z_t0, delta_logpz):
		logpz_t0 = self._base_dist.log_prob(z_t0)
		logpz_t1 = logpz_t0 - delta_logpz
		return logpz_t0, logpz_t1
	