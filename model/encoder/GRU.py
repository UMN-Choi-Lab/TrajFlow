import torch
import torch.nn as nn

class GRU(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(GRU, self).__init__()
		#self.gru = nn.GRU(input_dim + 2, hidden_dim, num_layers, batch_first=True)
		self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

	def forward(self, t, x):
		imputed = self._forward_imputation(t, x)
		#imputed = x	
		embedding, _ = self.gru(imputed)
		return embedding[:, -1, :]
	
	def _forward_imputation(self, t, x): # TODO: this is bad
		batch_size, seq_len, _ = x.shape
		mask = torch.isnan(x).any(dim=-1).float().unsqueeze(-1)

		t_deltas = torch.zeros_like(t)
		t_deltas[1:] = t[1:] - t[:-1]
		t_expanded = t_deltas.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1).clone()

		prev = [self._prev_map(mask[batch].squeeze(), seq_len) for batch in range(batch_size)]

		imputed = x.clone()
		for batch in range(batch_size):
			for idx, prev_idx in prev[batch].items():
				if prev_idx == -1:
					if mask[batch][idx]:  # No previous valid value
						imputed[batch, idx] = torch.zeros_like(imputed[batch, idx])
				else:
					if mask[batch][idx]:
						imputed[batch, idx] = x[batch, prev_idx]
					t_expanded[batch, idx] = t[idx] - t[prev_idx]

		imputed = torch.cat([imputed, mask, t_expanded], dim=-1)
		return imputed

	def _prev_map(self, mask, seq_len):
		p = -1
		prev_map = {} # TODO: list of size seq_len
		for i in range(seq_len):
			prev_map[i] = p
			if mask[i] == 0:
				p = i
		return prev_map
	
	def _next_map(self, mask, seq_len):
		n = 0
		next_map = {}
		for i in range(seq_len  - 1, -1, -1):
			if mask[i] == 0:
				n = i
			else:
				next_map[i] = n
		return next_map