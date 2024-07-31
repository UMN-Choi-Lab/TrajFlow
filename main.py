import torch
from InD import InD
from model.TrajFlow import TrajFlow,  CausalEnocder, Flow
from train import train
from evaluate import evaluate
from visualize import visualize

# TODO: these should be arg parsed
should_train = False
should_evaluate = False
should_visualize = True

ind = InD(
    root="data",
    train_ratio=0.7, 
    train_batch_size=64, 
    test_batch_size=1,
    missing_rate=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
traj_cnf = TrajFlow(
    seq_len=100, 
    input_dim=2, 
    feature_dim=5, 
    embedding_dim=128,
    hidden_dims=(512,512,512,512),
    causal_encoder=CausalEnocder.GRU,
    flow=Flow.CNF).to(device)

if should_train:
    train(
        observation_site=ind.observation_site8,
        model=traj_cnf,
        epochs=100,
        lr=1e-3,
        weight_decay=0,#1e-5,
        gamma=0.999,
        verbose=True)

traj_cnf.load_state_dict(torch.load('traj_cnf.pt'))

if should_evaluate:
    evaluate(
        observation_site=ind.observation_site8,
        model=traj_cnf,
        num_samples=1)

if should_visualize:
    visualize(
        observation_site=ind.observation_site8,
        model=traj_cnf,
        num_samples=10,
        steps=100,
        prob_threshold=0.001,
        output_dir='visualization') 
