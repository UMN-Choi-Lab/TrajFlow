import random
import time
import wandb
import torch
from InD import InD
from model.TrajFlow import TrajFlow,  CausalEnocder, Flow
from train import train
from evaluate import evaluate
from visualize import visualize

should_train = True
should_serialize = False
should_evaluate = True
should_visualize = False
verbose = True

ind_train = InD(
	root="data",
	train_ratio=0.6, 
	train_batch_size=64, 
	test_batch_size=1,
	missing_rate=0)
ind_train.observation_site1

with wandb.init() as run:
	run.config.setdefaults({
		'encoder': 'GRU',
		'flow': 'DNF',
		'masked_data_ratio': 0.5,
		'seed': random.randint(0, 2**32 - 1)
	})
	torch.manual_seed(run.config.seed)

	ind = InD(
		root="data",
		train_ratio=0.6, 
		train_batch_size=64, 
		test_batch_size=1,
		missing_rate=run.config.masked_data_ratio)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	traj_flow = TrajFlow(
		seq_len=100, 
		input_dim=2, 
		feature_dim=5, 
		embedding_dim=512,
		hidden_dims=(512,512,512,512),
		causal_encoder=CausalEnocder[run.config.encoder],
		flow=Flow[run.config.flow]).to(device)
	num_parameters = sum(p.numel() for p in traj_flow.parameters() if p.requires_grad)
	wandb.log({'parameters': num_parameters})

	start_time = time.time()

	total_loss = []
	if should_train:
		total_loss = train(
			observation_site=ind.observation_site1,
			model=traj_flow,
			epochs=25,#100,
			lr=1e-3,
			weight_decay=0,#1e-5,
			gamma=0.999,
			verbose=verbose,
			device=device)
		
	end_time = time.time()
	runtime = end_time - start_time
	wandb.log({'runtime': runtime})
		
	for loss in total_loss:
		wandb.log({'loss': loss})
			
	if should_serialize:
		model_name = f'traj_flow_{run.config.encoder}_{run.config.flow}_{run.config.masked_data_ratio}_{run.config.seed}.pt'
		if should_train:
			torch.save(traj_flow.state_dict(), model_name)
		traj_flow.load_state_dict(torch.load(model_name))

	if should_evaluate:
		rmse, crps = evaluate(
			observation_site=ind.observation_site1,
			model=traj_flow,
			num_samples=100,
			device=device)
		
		if verbose:
			print(f'rmse: {rmse}')
			print(f'crps: {crps}')
		wandb.log({'rmse': rmse, 'crps': crps})

	if should_visualize:
		visualize(
			observation_site=ind.observation_site1,
			model=traj_flow,
			num_samples=10,
			steps=100,
			prob_threshold=0.001,
			output_dir='visualization',
			device=device)
