import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(observation_site, model):
    model.eval()

    with torch.no_grad():
        mse = 0
        count = 0

        for inputs, feature in observation_site.test_loader:
            test_input = inputs[:, :100, ...].to(device)
            test_feature = feature[:, :100, ...].to(device)
            test_target = inputs[:, 100:, ...].to(device)
            _, samples, _ = model.sample(test_input, test_feature)
            mse += F.mse_loss(test_target, samples, reduction="mean")
            count += 1

        mse /= count
        print(f'Reconstruction Error: {mse}')