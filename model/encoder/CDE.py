import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

import numpy as np # natural cubic spline uses this... can we just use torch? I think we can use torch.empty

class NaturalCubicSpline:
	"""Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.

	Example:
		times = torch.linspace(0, 1, 7)
		# (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
		X = torch.rand(2, 1, 7, 3)
		coeffs = natural_cubic_spline_coeffs(times, X)
		# ...at this point you can save the coeffs, put them through PyTorch's Datasets and DataLoaders, etc...
		spline = NaturalCubicSpline(times, coeffs)
		t = torch.tensor(0.4)
		# will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
		out = spline.derivative(t)
	"""

	def __init__(self, times, path, **kwargs):
		"""
		Arguments:
			times: As was passed as an argument to natural_cubic_spline_coeffs.
			coeffs: As returned by natural_cubic_spline_coeffs.
		"""
		super(NaturalCubicSpline, self).__init__(**kwargs)

		# as we're typically computing derivatives, we store the multiples of these coefficients (c, d) that are more useful
		self._times = times
		a, b, two_c, three_d = self._coefficients(times, path.transpose(-1, -2))
		self._a = a.transpose(-1, -2)
		self._b = b.transpose(-1, -2)
		self._two_c = two_c.transpose(-1, -2)
		self._three_d = three_d.transpose(-1, -2)
		
	def _coefficients(self, times, path):
		# path should be a tensor of shape (..., length)
		# Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

		length = path.size(-1)

		if length < 2:
			# In practice this should always already be caught in __init__.
			raise ValueError("Must have a time dimension of size at least 2.")
		elif length == 2:
			a = path[..., :1]
			b = (path[..., 1:] - path[..., :1]) / (times[..., 1:] - times[..., :1])
			two_c = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
			three_d = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
		else:
			# Set up some intermediate values
			time_diffs = times[1:] - times[:-1]
			time_diffs_reciprocal = time_diffs.reciprocal()
			time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
			three_path_diffs = 3 * (path[..., 1:] - path[..., :-1])
			six_path_diffs = 2 * three_path_diffs
			path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

			# Solve a tridiagonal linear system to find the derivatives at the knots
			system_diagonal = torch.empty(length, dtype=path.dtype, device=path.device)
			system_diagonal[:-1] = time_diffs_reciprocal
			system_diagonal[-1] = 0
			system_diagonal[1:] += time_diffs_reciprocal
			system_diagonal *= 2
			system_rhs = torch.empty_like(path)
			system_rhs[..., :-1] = path_diffs_scaled
			system_rhs[..., -1] = 0
			system_rhs[..., 1:] += path_diffs_scaled
			knot_derivatives = self._tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal,
												  time_diffs_reciprocal)

			# Do some algebra to find the coefficients of the spline
			a = path[..., :-1]
			b = knot_derivatives[..., :-1]
			two_c = (six_path_diffs * time_diffs_reciprocal
				- 4 * knot_derivatives[..., :-1]
				- 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
			three_d = (-six_path_diffs * time_diffs_reciprocal
				+ 3 * (knot_derivatives[..., :-1]
					+ knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

		return a, b, two_c, three_d

	def _tridiagonal_solve(self, b, A_upper, A_diagonal, A_lower):
		"""Solves a tridiagonal system Ax = b.

		The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
		and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
		of size (k, k), with entries:

		D[0] U[0]
		L[0] D[1] U[1]
			L[1] D[2] U[2]                     0
				L[2] D[3] U[3]
					.    .    .
						.      .      .
							.        .        .
								L[k - 3] D[k - 2] U[k - 2]
		   0                            L[k - 2] D[k - 1] U[k - 1]
											L[k - 1]   D[k]

		Arguments:
			b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
			A_upper: A tensor of shape (..., k - 1).
			A_diagonal: A tensor of shape (..., k).
			A_lower: A tensor of shape (..., k - 1).

		Returns:
			A tensor of shape (..., k), corresponding to the x solving Ax = b

		Warning:
			This implementation isn't super fast. You probably want to cache the result, if possible.
		"""

		# This implementation is very much written for clarity rather than speed.
		A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
		A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
		A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)

		channels = b.size(-1)

		new_b = np.empty(channels, dtype=object)
		new_A_diagonal = np.empty(channels, dtype=object)
		outs = np.empty(channels, dtype=object)

		new_b[0] = b[..., 0]
		new_A_diagonal[0] = A_diagonal[..., 0]
		for i in range(1, channels):
			w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
			new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
			new_b[i] = b[..., i] - w * new_b[i - 1]

		outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
		for i in range(channels - 2, -1, -1):
			outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

		return torch.stack(outs.tolist(), dim=-1)

	def _interpret_t(self, t):
		maxlen = self._b.size(-2) - 1
		index = (t > self._times).sum() - 1
		index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
		# will never access the last element of self._times; this is correct behaviour
		fractional_part = t - self._times[index]
		return fractional_part, index

	def evaluate(self, t):
		"""Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
		fractional_part, index = self._interpret_t(t)
		inner = 0.5 * self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part / 3
		inner = self._b[..., index, :] + inner * fractional_part
		return self._a[..., index, :] + inner * fractional_part

	def derivative(self, t):
		"""Evaluates the derivative of the natural cubic spline at a point t, which should be a scalar tensor."""
		fractional_part, index = self._interpret_t(t)
		inner = self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
		deriv = self._b[..., index, :] + inner * fractional_part
		return deriv


class CDEFunc(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers=3): # TODO: use num_layers
		super(CDEFunc, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		
		dim_list = [hidden_dim] * num_layers + [input_dim * hidden_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			#if i < len(dim_list) - 2:
			layers.append(nn.LayerNorm(dim_list[i + 1]))
			layers.append(nn.Softplus())
		self.mlp = nn.Sequential(*layers)
		#self.norm = nn.LayerNorm(dim_list[-1])
	
	def forward(self, x):
		x = self.mlp(x)
		#x = self.norm(x)
		#x = x.tanh()
		x = x.view(*x.shape[:-1], self.hidden_dim, self.input_dim)
		return x
	

class VectorField(torch.nn.Module):
	def __init__(self, dX_dt, f):
		super(VectorField, self).__init__()
		self.dX_dt = dX_dt
		self.f = f

	def forward(self, t, z):
		dX_dt = self.dX_dt(t)
		f = self.f(z)
		out = (f @ dX_dt.unsqueeze(-1)).squeeze(-1)
		return out
	

class CDE(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(CDE, self).__init__()
		self.embed = torch.nn.Linear(input_dim, hidden_dim)
		self.f = CDEFunc(input_dim, hidden_dim, num_layers)

	def forward(self, t, x):
		print(t.shape)
		print(x.shape)
		# TODO: We need to append t as a channel

		spline = NaturalCubicSpline(t, x)
		vector_field = VectorField(dX_dt=spline.derivative, f=self.f)
		z0 = self.embed(spline.evaluate(t[0]))
		out = odeint_adjoint(vector_field, z0, t, method='dopri5', atol=1e-5, rtol=1e-5)
		embedding = out[1]
		return embedding