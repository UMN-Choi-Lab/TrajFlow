import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
	
class ODEFunc(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims):
		super(ODEFunc, self).__init__()
		dim_list = [input_dim + condition_dim] + list(hidden_dims) + [input_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i] + 2, dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.LayerNorm(dim_list[i + 1]))
		self.layers = nn.ModuleList(layers)
		self.initial_norm = nn.LayerNorm(dim_list[0])
		self.condition = None

	def _z_dot(self, t, z):
		positional_encoding = (torch.cumsum(torch.ones_like(z)[:, :, 0], 1) / z.shape[1]).unsqueeze(-1)
		time_encoding = t.expand(z.shape[0], z.shape[1], 1)
		condition = self.condition.unsqueeze(1).expand(-1, z.shape[1], -1)
		z_dot = torch.cat([z, condition], dim=-1)
		z_dot = self.initial_norm(z_dot)
		for i in range(0, len(self.layers), 2):
			zpt_cat = torch.cat([z_dot, time_encoding, positional_encoding], dim=-1)
			z_dot = self.layers[i](zpt_cat)
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


class CNF(torch.nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims):
		super(CNF, self).__init__()
		self.time_derivative = ODEFunc(input_dim, condition_dim, hidden_dims)

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
	