import torch.nn as nn

class GRU(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(GRU, self).__init__()
		self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

	def forward(self, t, x): # t could be used for delta-t GRU
		embedding, _ = self.gru(x)
		return embedding[:, -1, :]
	