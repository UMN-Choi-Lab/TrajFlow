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

def evaluate(observation_site, model, num_samples, device):
    model.eval()

    with torch.no_grad():
        nll_sum = 0
        rmse_sum = 0
        crps_sum = 0
        count = 0

        for inputs, feature in observation_site.test_loader:
            test_input = inputs[:, :100, ...].to(device)
            test_feature = feature[:, :100, ...].to(device)
            test_target = inputs[:, 100:, ...].to(device)

            # NLL same as training loss
            z_t0, delta_logpz = model(test_input, test_target, test_feature)
            logpz_t0, logpz_t1 = model.log_prob(z_t0, delta_logpz)
            nll_sum += -torch.mean(logpz_t1)

            # sample based evaluation
            _, samples, _ = model.sample(test_input, test_feature, num_samples)
            rmse_sum += rmse(test_target, samples)
            crps_sum += crps(test_target, samples)
            count += 1

        rmse_score = rmse_sum / count
        crps_score = crps_sum / count
        nll_score = nll_sum / count

    return rmse_score, crps_score, nll_score
