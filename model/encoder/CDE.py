import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from model.layers.Spline import NaturalCubicSpline

class CDEFunc(nn.Module):
	def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=3): # TODO: use num_layers
		super(CDEFunc, self).__init__()
		self.input_dim = input_dim
		self.embedding_dim = embedding_dim
		
		dim_list = [embedding_dim] + [hidden_dim] * num_layers + [input_dim * embedding_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.ReLU())
		self.mlp = nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.mlp(x)
		x = x.tanh()
		x = x.view(*x.shape[:-1], self.embedding_dim, self.input_dim)
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
	def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers):
		super(CDE, self).__init__()
		self.embed = torch.nn.Linear(input_dim + 1, embedding_dim)
		self.f = CDEFunc(input_dim + 1, embedding_dim, hidden_dim, num_layers)

	def forward(self, t, x):
		spline = NaturalCubicSpline(t, x)
		vector_field = VectorField(dX_dt=spline.derivative, f=self.f)
		z0 = self.embed(spline.evaluate(t[0]))
		out = odeint_adjoint(vector_field, z0, t, method='dopri5', atol=1e-5, rtol=1e-5)
		embedding = out[-1]
		return embedding