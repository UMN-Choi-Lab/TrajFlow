import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchdiffeq import odeint_adjoint

def reduce_tensor(tensor, world_size=None):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	if world_size is None:
		world_size = dist.get_world_size()

	rt /= world_size
	return rt

class MovingBatchNormNd(nn.Module):
	def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True, sync=False):
		super(MovingBatchNormNd, self).__init__()
		self.num_features = num_features
		self.sync = sync
		self.affine = affine
		self.eps = eps
		self.decay = decay
		self.bn_lag = bn_lag
		self.register_buffer('step', torch.zeros(1))
		if self.affine:
			self.weight = nn.Parameter(torch.Tensor(num_features))
			self.bias = nn.Parameter(torch.Tensor(num_features))
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		self.reset_parameters()

	@property
	def shape(self):
		raise NotImplementedError

	def reset_parameters(self):
		self.running_mean.zero_()
		self.running_var.fill_(1)
		if self.affine:
			self.weight.data.zero_()
			self.bias.data.zero_()

	def forward(self, x, logpx=None, reverse=False):
		if reverse:
			return self._reverse(x, logpx)
		else:
			return self._forward(x, logpx)

	def _forward(self, x, logpx=None):
		num_channels = x.size(-1)
		used_mean = self.running_mean.clone().detach()
		used_var = self.running_var.clone().detach()

		if self.training:
			# compute batch statistics
			x_t = x.transpose(0, 1).reshape(num_channels, -1)
			batch_mean = torch.mean(x_t, dim=1)

			if self.sync:
				batch_ex2 = torch.mean(x_t**2, dim=1)
				batch_mean = reduce_tensor(batch_mean)
				batch_ex2 = reduce_tensor(batch_ex2)
				batch_var = batch_ex2 - batch_mean**2
			else:
				batch_var = torch.var(x_t, dim=1)

			# moving average
			if self.bn_lag > 0:
				used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
				used_mean /= (1. - self.bn_lag**(self.step[0] + 1))
				used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
				used_var /= (1. - self.bn_lag**(self.step[0] + 1))

			# update running estimates
			self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
			self.running_var -= self.decay * (self.running_var - batch_var.data)
			self.step += 1

		# perform normalization
		used_mean = used_mean.view(*self.shape).expand_as(x)
		used_var = used_var.view(*self.shape).expand_as(x)

		y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))

		if self.affine:
			weight = self.weight.view(*self.shape).expand_as(x)
			bias = self.bias.view(*self.shape).expand_as(x)
			y = y * torch.exp(weight) + bias

		if logpx is None:
			return y
		else:
			logpy = self._logdetgrad(x, used_var).sum(-1, keepdim=True)
			return y, logpx - logpy

	def _reverse(self, y, logpy=None):
		used_mean = self.running_mean
		used_var = self.running_var

		if self.affine:
			weight = self.weight.view(*self.shape).expand_as(y)
			bias = self.bias.view(*self.shape).expand_as(y)
			y = (y - bias) * torch.exp(-weight)

		used_mean = used_mean.view(*self.shape).expand_as(y)
		used_var = used_var.view(*self.shape).expand_as(y)
		x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

		if logpy is None:
			return x
		else:
			return x, logpy + self._logdetgrad(x, used_var).sum(-1, keepdim=True)

	def _logdetgrad(self, x, used_var):
		logdetgrad = -0.5 * torch.log(used_var + self.eps)
		if self.affine:
			weight = self.weight.view(*self.shape).expand(*x.size())
			logdetgrad += weight
		return logdetgrad

	def __repr__(self):
		return (
			'{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
			' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)
		)


def stable_var(x, mean=None, dim=1):
	if mean is None:
		mean = x.mean(dim, keepdim=True)
	mean = mean.view(-1, 1)
	res = torch.pow(x - mean, 2)
	max_sqr = torch.max(res, dim, keepdim=True)[0]
	var = torch.mean(res / max_sqr, 1, keepdim=True) * max_sqr
	var = var.view(-1)
	# change nan to zero
	var[var != var] = 0
	return var


class MovingBatchNorm1d(MovingBatchNormNd):
	@property
	def shape(self):
		return [1, -1]

	def forward(self, x, logpx=None, reverse=False):
		ret = super(MovingBatchNorm1d, self).forward(
				x, logpx=logpx, reverse=reverse)
		return ret


class ConcatSquashLinear(nn.Module):
	def __init__(self, dim_in, dim_out, dim_c):
		super(ConcatSquashLinear, self).__init__()
		self._layer = nn.Linear(dim_in, dim_out)
		self._hyper_bias = nn.Linear(dim_c, dim_out, bias=False)
		self._hyper_gate = nn.Linear(dim_c, dim_out)

	def forward(self, context, x):
		gate = torch.sigmoid(self._hyper_gate(context))
		bias = self._hyper_bias(context)
		ret = self._layer(x) * gate + bias
		return ret
	
class ODEFunc(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims):
		super(ODEFunc, self).__init__()
		dim_list = [input_dim] + list(hidden_dims) + [input_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			#layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			layers.append(ConcatSquashLinear(dim_list[i], dim_list[i + 1], condition_dim + 2))
			#if i < len(dim_list) - 2:
				#layers.append(nn.LayerNorm(dim_list[i + 1]))
				#layers.append(nn.Softplus())
				#layers.append(nn.ReLU())
				#layers.append(nn.LayerNorm(dim_list[i + 1]))
		self.layers = nn.ModuleList(layers)
		#self.mlp = nn.Sequential(*layers)
		#self.initial_norm = nn.BatchNorm1d(condition_dim)
		#self.batch_norm = nn.BatchNorm1d(input_dim)
		self.condition = None

	def _z_dot(self, t, z, condition):
		positional_encoding = (torch.cumsum(torch.ones_like(z)[:, :, 0], 1) / z.shape[1]).unsqueeze(-1)
		time_encoding = t.expand(z.shape[0], z.shape[1], 1)
		#condition = self.initial_norm(self.condition)
		condition = condition.unsqueeze(1).expand(-1, z.shape[1], -1)
		tpc = torch.cat([positional_encoding, time_encoding, condition], dim=-1)
		#condition.tanh()
		z_dot = z#torch.cat([z, positional_encoding, time_encoding, condition], dim=-1)
	
		#z_dot = self.initial_norm(z_dot)
		for l, layer in enumerate(self.layers):
			z_dot = layer(tpc, z_dot)
			if l < len(self.layers) - 1:
				z_dot = F.softplus(z_dot)
		#z_dot = self.mlp(z_dot)

		#batch_size, seq_len, num_features = z_dot.size()
		#z_dot = z_dot.view(batch_size * seq_len, num_features)
		#z_dot = self.batch_norm(z_dot)
		#z_dot = z_dot.view(batch_size, seq_len, num_features)

		#z_dot = z_dot.tanh()
		return z_dot
	
	def _jacobian_trace(seld, z_dot, z):
		batch_size, seq_len, dim = z.shape
		trace = torch.zeros(batch_size, seq_len, device=z.device)
		for i in range(dim):
			trace += torch.autograd.grad(z_dot[:, :, i].sum(), z, create_graph=True)[0][:, :, i]
		return trace
	
	def forward(self, t, states):
		z = states[0]
		condition = states[2]

		with torch.set_grad_enabled(True):
			z.requires_grad_(True)
			t.requires_grad_(True)
			condition.requires_grad_(True)
			z_dot = self._z_dot(t, z, condition)
			divergence = self._jacobian_trace(z_dot, z)

		return z_dot, -divergence, torch.zeros_like(condition).requires_grad_(True)


class CNF(torch.nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims):
		super(CNF, self).__init__()
		self.time_derivative = ODEFunc(input_dim, condition_dim, hidden_dims)
		self.condition_norm = nn.LayerNorm(condition_dim)
		self.pre_flow_norm = MovingBatchNorm1d(input_dim)
		self.post_flow_norm = MovingBatchNorm1d(input_dim)

	def forward(self, z, condition, delta_logpz=None, integration_times=None, reverse=False):
		if delta_logpz is None:
			delta_logpz = torch.zeros(z.shape[0], z.shape[1], 1).to(z)
		if integration_times is None:
			integration_times = torch.tensor([0.0, 1.0]).to(z)
		if reverse:
			integration_times = torch.flip(integration_times, [0])

		#self.time_derivative.condition = self.initial_norm(condition)
		#state = odeint_adjoint(self.time_derivative, (z, delta_logpz), integration_times, method='dopri5', atol=1e-5, rtol=1e-5)
		condition = self.condition_norm(condition)
		
		z, delta_logpz = self.pre_flow_norm(z, delta_logpz, reverse)
		
		state = odeint_adjoint(self.time_derivative, (z, delta_logpz, condition), integration_times, method='dopri5', atol=1e-5, rtol=1e-5)

		if len(integration_times) == 2:
			state = tuple(s[1] for s in state)
		z, delta_logpz, condition = state

		z, delta_logpz = self.post_flow_norm(z, delta_logpz, reverse)
		return z, delta_logpz.squeeze(-1)
	