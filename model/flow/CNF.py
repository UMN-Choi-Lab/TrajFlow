import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from model.layers.MovingBatchNorm import MovingBatchNorm1d
from model.layers.SquashLinear import ConcatSquashLinear
	
class ODEFunc(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims, marginal):
		super(ODEFunc, self).__init__()

		self.marginal = marginal
		self.epsilon = None

		temporal_context_dim = 2 if marginal else 1
		dim_list = [input_dim] + list(hidden_dims) + [input_dim]
		
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(ConcatSquashLinear(dim_list[i], dim_list[i + 1], condition_dim + temporal_context_dim))
		self.layers = nn.ModuleList(layers)

	def _z_dot(self, t, z, condition):		
		if self.marginal:
			condition = condition.unsqueeze(1).expand(-1, z.shape[1], -1)
			time_encoding = t.expand(z.shape[0], z.shape[1], 1)
			#positional_encoding = (torch.cumsum(torch.ones_like(z)[:, :, 0], 1) / z.shape[1]).unsqueeze(-1)
			positional_encoding = torch.cumsum(torch.ones_like(z)[:, :, 0], 1).unsqueeze(-1)
			context = torch.cat([positional_encoding, time_encoding, condition], dim=-1)
		else:
			time_encoding = t.expand(z.shape[0], 1)
			context = torch.cat([time_encoding, condition], dim=-1)

		z_dot = z
		for l, layer in enumerate(self.layers):
			z_dot = layer(context, z_dot)
			if l < len(self.layers) - 1:
				z_dot = F.tanh(z_dot)
		return z_dot
	
	def _gaussian_noise(self, z):
		noise = torch.randn_like(z).to(z)
		self.epsilon = noise
	
	def _rademacher_noise(self, z):
		random_bits = torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype)
		noise = 2 * random_bits - 1
		self.epsilon = noise
		
	def _hutchinson_estimator(self, z_dot, z):
		e = self.epsilon
		z_dot_e = torch.autograd.grad(z_dot, z, grad_outputs=e, create_graph=True)[0]
		trace_estimate = torch.sum(z_dot_e * e, dim=-1)
		return trace_estimate
	
	def _jacobian_trace_joint(self, z_dot, z): # might need hutchson estimator here for efficency
		return self._hutchinson_estimator(z_dot, z)
		trace = 0.0
		for i in range(z_dot.shape[1]):
			trace += torch.autograd.grad(z_dot[:, i].sum(), z, create_graph=True)[0][:, i]
		return trace

	def _jacobian_trace_marginal(self, z_dot, z):
		batch_size, seq_len, dim = z.shape
		trace = torch.zeros(batch_size, seq_len, device=z.device)
		for i in range(dim):
			trace += torch.autograd.grad(z_dot[:, :, i].sum(), z, create_graph=True)[0][:, :, i]
		return trace
	
	def _jacobian_trace(self, z_dot, z):
		return self._jacobian_trace_marginal(z_dot, z) if self.marginal else self._jacobian_trace_joint(z_dot, z)
	
	def forward(self, t, states):
		z = states[0]
		condition = states[2]

		with torch.set_grad_enabled(True):
			t.requires_grad_(True)
			for state in states:
				state.requires_grad_(True)
			z_dot = self._z_dot(t, z, condition)
			divergence = self._jacobian_trace(z_dot, z)

		return z_dot, -divergence, torch.zeros_like(condition).requires_grad_(True)


class CNF(torch.nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims, marginal):
		super(CNF, self).__init__()
		self.marginal = marginal
		self.time_derivative = ODEFunc(input_dim, condition_dim, hidden_dims, marginal)
		self.condition_norm = nn.LayerNorm(condition_dim)
		self.n1 = MovingBatchNorm1d(input_dim)
		self.n2 = MovingBatchNorm1d(input_dim)

	def forward(self, z, condition, delta_logpz=None, integration_times=None, reverse=False):
		if delta_logpz is None:
			delta_logpz = torch.zeros(z.shape[0], z.shape[1], 1).to(z) if self.marginal else torch.zeros(z.shape[0], 1).to(z)
		if integration_times is None:
			integration_times = torch.tensor([0.0, 1.0]).to(z)
		if reverse:
			integration_times = torch.flip(integration_times, [0])

		condition = self.condition_norm(condition) #no norm? lets let the encoder handle this if needed

		if not self.marginal: # if not marginal produce noise for hutchinson estimation
			self.time_derivative._rademacher_noise(z)
		
		z, delta_logpz = self.n1(z, delta_logpz, reverse) if not reverse else self.n2(z, delta_logpz, reverse)
		state = odeint_adjoint(self.time_derivative, (z, delta_logpz, condition), integration_times, method='dopri5', atol=1e-5, rtol=1e-5)
		z, delta_logpz, condition = tuple(s[1] for s in state) if len(integration_times) == 2 else state
		z, delta_logpz = self.n2(z, delta_logpz, reverse) if not reverse else self.n1(z, delta_logpz, reverse)
		return z, delta_logpz.squeeze(-1)
	