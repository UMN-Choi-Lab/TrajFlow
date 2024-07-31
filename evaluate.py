import torch
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def crps(y_true, samples):
    abs_diff = torch.abs(samples - y_true.unsqueeze(1))
    sample_diff = torch.abs(samples.unsqueeze(2) - samples.unsqueeze(1))
    crps = abs_diff.mean(dim=1) - 0.5 * sample_diff.mean(dim=(1, 2))
    return crps.mean()

def crps_torch(y_true, samples):
    # y_true: [batch, seq_len, features]
    # samples: [batch, num_samples, seq_len, features]
    num_samples = samples.shape[1]
    
    mae = torch.abs(samples - y_true.unsqueeze(1)).mean(dim=1)
    
    sorted_samples, _ = torch.sort(samples, dim=1)
    diff = sorted_samples[:, 1:] - sorted_samples[:, :-1]
    
    weight = torch.arange(1, num_samples, device=samples.device) * torch.arange(num_samples - 1, 0, -1, device=samples.device)
    weight = weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) / (num_samples ** 2)
    
    crps = mae - (diff * weight).sum(dim=1)
    
    return crps.mean()

def rmse(y_true, samples):
    mse = F.mse_loss(y_true, samples.mean(dim=1), reduction="mean")
    rmse = torch.sqrt(mse)
    return rmse

def evaluate(observation_site, model, num_samples=100):
    model.eval()

    with torch.no_grad():
        rmse_sum = 0
        crps_sum = 0
        crps2_sum = 0
        count = 0

        for inputs, feature in observation_site.test_loader:
            test_input = inputs[:, :100, ...].to(device)
            test_feature = feature[:, :100, ...].to(device)
            test_target = inputs[:, 100:, ...].to(device)

            samples_list = []
            for _ in range(num_samples):
                _, sample, _ = model.sample(test_input, test_feature)
                # print('----------')
                # print(sample.shape)
                # print('----------')
                samples_list.append(sample)
            samples = torch.stack(samples_list, dim=1)

            # print(test_target.shape)
            # print(samples.shape)
            # print('----------')
            # print(test_target[0, :])
            # print('----------')
            # print(samples[0, 0, :])
            # print('----------')

            rmse_sum += rmse(test_target, samples)

            crps_sum += crps(test_target, samples)

            #crps2_sum += crps_torch(test_target, samples)

            count += 1

        rmse_score = rmse_sum / count
        crps_score = crps_sum / count
        crps2_score = crps2_sum / count
        print(f'RMSE: {rmse_score}')
        print(f'CRPS: {crps_score}')
        print(f'CRPS2: {crps2_score}')

    return rmse_score, crps_score