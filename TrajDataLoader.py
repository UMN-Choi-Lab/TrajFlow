import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrajDataset(Dataset):
    def __init__(self, input, feature):
        assert input.shape[0] == feature.shape[0]
        self.input = input
        self.feature = feature
        self.data_size = input.size(0)

    def __getitem__(self, index):
        return self.input[index], self.feature[index]

    def __len__(self):
        return self.data_size


# TODO: simplify this
def normalization(raw_input, x_min, x_max, y_min, y_max):
    def apply_normalize_x(x): return (x-x_min)/(x_max-x_min) if x != -1 else -1
    def apply_normalize_y(y): return (y-y_min)/(y_max-y_min) if y != -1 else -1

    raw_input[:, :, 0] = np.vectorize(apply_normalize_x)(raw_input[:, :, 0])
    raw_input[:, :, 1] = np.vectorize(apply_normalize_y)(raw_input[:, :, 1])
    return raw_input   

def load_data(input, features, train_ratio, train_batch_size, test_batch_size):
    boundaries=[30, 81, -61, -5]
    feature_boundaries = [[0, 360], [-10, 10], [-10, 10], [-5, 5], [-5, 5]]

    input = normalization(input, boundaries[0], boundaries[1], boundaries[2], boundaries[3])
    features = (features - feature_boundaries[:, 0]) / (feature_boundaries[:, 1] - feature_boundaries[:, 0])

    randidx = np.random.permutation(input.shape[0])
    n_data = len(randidx)
    n_train = int(n_data * train_ratio)
    n_test = n_data - n_train

    train_idx = randidx[:n_train]
    test_idx = randidx[n_train:(n_train+n_test)]

    train_input = torch.FloatTensor(input[train_idx])
    test_input = torch.FloatTensor(input[test_idx])

    train_feature = torch.FloatTensor(features[train_idx])
    test_feature = torch.FloatTensor(features[test_idx])

    train_data = TrajDataset(train_input, train_feature)
    test_data = TrajDataset(test_input, test_feature)

    train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=True)
    return train_loader, test_loader 
