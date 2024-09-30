import copy
import torch
import torch.nn as nn
from torch.distributions import Normal

class SequentialFlow(nn.Sequential):
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians
    

class RunningAverageBatchNorm(nn.Module):
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.view(-1, x.shape[-1]).mean(0)
            self.batch_var = x.view(-1, x.shape[-1]).var(0)

            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)
    

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=0):
        super().__init__()

        self.register_buffer("mask", mask)

        s_net = [nn.Linear(input_size + 1 + cond_label_size, hidden_size,)]
        for _ in range(n_hidden):
            s_net += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.ReLU(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        self.t_net = copy.deepcopy(self.s_net)

    def forward(self, x, y=None):
        mx = x * self.mask
        y = y.unsqueeze(1).expand(-1, mx.shape[1], -1) if y is not None else None

        index = (torch.cumsum(torch.ones_like(x)[:, :, 0], 1) / x.shape[1]).unsqueeze(-1)

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, index, mx], dim=-1))
        t = self.t_net(mx if y is None else torch.cat([y, index, mx], dim=-1)) * (1 - self.mask)
        
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t
        log_abs_det_jacobian = log_s
        
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        mu = u * self.mask
        y = y.unsqueeze(1).expand(-1, mu.shape[1], -1) if y is not None else None

        index = (torch.cumsum(torch.ones_like(u)[:, :, 0], 1) / u.shape[1]).unsqueeze(-1)

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, index, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, index, mu], dim=-1)) * (1 - self.mask)
        
        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)
        log_abs_det_jacobian = -log_s
        
        return x, log_abs_det_jacobian


class DNF(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__()
        modules = []
        mask = torch.arange(input_size).float() % 2
        for _ in range(n_blocks):
            modules += [AffineCouplingLayer(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [RunningAverageBatchNorm(input_size)]

        self.net = SequentialFlow(*modules)
    
    def forward(self, z, condition, reverse=False):
        z, delta_logpz = self.net.inverse(z, condition) if reverse else self.net.forward(z, condition)
        return z, torch.sum(-delta_logpz, dim=-1) #Negative to match CNF formulation