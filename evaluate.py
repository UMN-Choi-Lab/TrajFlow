import math
import torch
import torch.nn.functional as F

def rmse(y_true, y_pred):
    mse = F.mse_loss(y_true.expand_as(y_pred), y_pred, reduction="mean")
    rmse = torch.sqrt(mse)
    return rmse

def crps(y_true, y_pred):
    num_samples = y_pred.shape[0]
    absolute_error = torch.mean(torch.abs(y_pred - y_true), dim=0)

    if num_samples == 1:
        return  torch.mean(absolute_error)

    y_pred, _ = torch.sort(y_pred, dim=0)
    empirical_cdf = torch.arange(num_samples, device=y_pred.device).view(-1, 1, 1) / num_samples
    b0 = torch.mean(y_pred, dim=0)
    b1 = torch.mean(y_pred * empirical_cdf, dim=0)

    crps = absolute_error + b0 - 2 * b1
    crps = torch.mean(crps)
    return crps

def min_ade(y_true, y_pred):
    num_samples, seq_len, _ = y_pred.shape
    ade = [1000 for _ in range(seq_len)]
    for i in range(num_samples):
        for j in range(seq_len):
            x_squared = (y_pred[i][j][0] - y_true[0][j][0]) ** 2
            y_squared = (y_pred[i][j][1] - y_true[0][j][1]) ** 2
            l2_distance = math.sqrt(x_squared + y_squared)
            ade[j] = min(ade[j], l2_distance)
    return sum(ade) / y_pred.shape[1]

def min_fde(y_true, y_pred):
    fde = torch.norm(y_pred[:,-1,:] - y_true[:,-1,:], dim=-1)
    return fde.min()

def evaluate(observation_site, model, num_samples, device):
    model.eval()

    with torch.no_grad():
        nll_sum = 0
        rmse_sum = 0
        crps_sum = 0
        min_ade_sum = 0
        min_fde_sum = 0
        count = 0

        for test_input, test_feature, test_target in observation_site.test_loader:
            test_input = test_input.to(device)
            test_feature = test_feature.to(device)
            test_target = test_target.to(device)

            # NLL same as training loss
            z_t0, delta_logpz = model(test_input, test_target, test_feature)
            logpz_t0, logpz_t1 = model.log_prob(z_t0, delta_logpz)
            nll_sum += -torch.mean(logpz_t1)

            # sample based evaluation
            _, samples, _ = model.sample(test_input, test_feature, num_samples)
            test_target = torch.tensor(observation_site.denormalize(test_target.cpu().numpy())).to(device)
            samples = torch.tensor(observation_site.denormalize(samples.cpu().numpy())).to(device)

            rmse_sum += rmse(test_target, samples)
            crps_sum += crps(test_target, samples)
            min_ade_sum += min_ade(test_target, samples)
            min_fde_sum += min_fde(test_target, samples)
            count += 1

        rmse_score = rmse_sum / count
        crps_score = crps_sum / count
        min_ade_score = min_ade_sum / count
        min_fde_score = min_fde_sum / count
        nll_score = nll_sum / count

    return rmse_score, crps_score, min_ade_score, min_fde_score, nll_score
