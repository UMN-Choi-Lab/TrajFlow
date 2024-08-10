import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from model.layers.Spline import NaturalCubicSpline
from model.layers.MovingBatchNorm import MovingBatchNorm1d
from model.layers.SquashLinear import SquashLinear

# class CDEFunc(nn.Module):
# 	def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=3): # TODO: use num_layers
# 		super(CDEFunc, self).__init__()
# 		self.input_dim = input_dim
# 		self.embedding_dim = embedding_dim
		
# 		dim_list = [embedding_dim] + [hidden_dim] * num_layers + [input_dim * embedding_dim]
# 		layers = []
# 		for i in range(len(dim_list) - 1):
# 			layers.append(SquashLinear(dim_list[i], dim_list[i + 1], 1))
# 		self.layers = nn.ModuleList(layers)

# 		#layers = []
# 		#for i in range(len(dim_list) - 1):
# 		#	layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
# 		#	if i < len(dim_list) - 2:
# 		#		layers.append(nn.Tanh())
# 				#layers.append(nn.ReLU())
# 		#self.mlp = nn.Sequential(*layers)
	
# 	def forward(self, t, x):
# 		#print(x.shape)
# 		#print(t.shape)
# 		#batch_size, seq_len, _ = x.shape
# 		t_expanded = t.unsqueeze(0)#t.unsqueeze(0).unsqueeze(-1)
# 		t_expanded = t_expanded.expand(x.shape[0], 1)#t_expanded.expand(batch_size, seq_len, 1)
# 		for l, layer in enumerate(self.layers):
# 			x = layer(t_expanded, x)
# 			if l < len(self.layers) - 1:
# 				x = F.tanh(x)
# 		#x = self.mlp(x)
# 		#x = x.tanh()
# 		x = x.view(*x.shape[:-1], self.embedding_dim, self.input_dim)
# 		return x
	

# class VectorField(torch.nn.Module):
# 	def __init__(self, dX_dt, f):
# 		super(VectorField, self).__init__()
# 		self.dX_dt = dX_dt
# 		self.f = f

# 	def forward(self, t, z):
# 		t.requires_grad_(True)
# 		z.requires_grad_(True)
# 		dX_dt = self.dX_dt(t)
# 		f = self.f(t, z)
# 		out = (f @ dX_dt.unsqueeze(-1)).squeeze(-1)
# 		return out
	

# class CDE(torch.nn.Module):
# 	def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers):
# 		super(CDE, self).__init__()
# 		self.embed = torch.nn.Linear(input_dim + 1, embedding_dim)
# 		self.f = CDEFunc(input_dim + 1, embedding_dim, hidden_dim, num_layers)
# 		self.n1 = MovingBatchNorm1d(input_dim + 1)
# 		self.n2 = MovingBatchNorm1d(embedding_dim)

# 	def forward(self, t, x):
# 		batch_size, seq_len, _ = x.shape
# 		t_expanded = t.unsqueeze(0).unsqueeze(-1)
# 		t_expanded = t_expanded.expand(batch_size, seq_len, 1)
# 		x = torch.cat([x, t_expanded], dim=-1)
# 		x = self.n1(x)
# 		spline = NaturalCubicSpline(t, x)
# 		vector_field = VectorField(dX_dt=spline.derivative, f=self.f)
# 		z0 = self.embed(spline.evaluate(t[0]))
# 		#z0 = self.n1(z0) # first norm here or pre embed?
# 		out = odeint_adjoint(vector_field, z0, t, method='dopri5', atol=1e-5, rtol=1e-5)
# 		embedding = self.n2(out[1]) # second norm here?
# 		return embedding

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
		batch_size, seq_len, _ = x.shape
		t_expanded = t.unsqueeze(0).unsqueeze(-1)
		t_expanded = t_expanded.expand(batch_size, seq_len, 1)
		x = torch.cat([x, t_expanded], dim=-1)
		spline = NaturalCubicSpline(t, x)
		vector_field = VectorField(dX_dt=spline.derivative, f=self.f)
		z0 = self.embed(spline.evaluate(t[0]))
		out = odeint_adjoint(vector_field, z0, t, method='dopri5', atol=1e-5, rtol=1e-5)
		embedding = out[-1]
		return embedding