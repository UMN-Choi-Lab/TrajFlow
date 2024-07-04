import torch
from InD import InD
from TrajCNF import TrajCNF
from train import train
from visualize import visualize

# TODO: these 2 should be arg parsed
should_train = True
should_visualize = True

ind = InD(
    root="data",
    train_ratio=0.7, 
    train_batch_size=64, 
    test_batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
traj_cnf = TrajCNF(
    seq_len=100, 
    input_dim=2, 
    feature_dim=5, 
    embedding_dim=128,
    hidden_dims=(130,65)).to(device)

if should_train:
    train(
        observation_site=ind.observation_site8,
        model=traj_cnf,
        epochs=100,
        lr=1e-3,
        gamma=0.999,
        verbose=True)

traj_cnf.load_state_dict(torch.load('traj_cnf.pt'))

if should_visualize:
    visualize(
        observation_site=ind.observation_site8,
        model=traj_cnf,
        num_samples=10,
        steps=100,
        output_dir='visualization') 
