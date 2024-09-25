import random
import time
import wandb
import torch
from datasets.InD import InD
from model.TrajFlow import TrajFlow,  CausalEnocder, Flow
from train import train
from evaluate import evaluate
from visualize import visualize

should_train = False
should_serialize = True
should_evaluate = False
should_visualize = True
verbose = True
simple_visualization = True

with wandb.init() as run:
	run.config.setdefaults({
		'seed': random.randint(0, 2**32 - 1),
		'encoder': 'CDE',
		'flow': 'CNF',
		'masked_data_ratio': 0
	})
	torch.manual_seed(run.config.seed)

	ind = InD(
		root="data",
		train_ratio=0.75, 
		train_batch_size=64, 
		test_batch_size=1,
		missing_rate=run.config.masked_data_ratio)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(device)

	traj_flow = TrajFlow(
		seq_len=100, 
		input_dim=2, 
		feature_dim=5, 
		embedding_dim=128,
		hidden_dim=512,
		causal_encoder=CausalEnocder[run.config.encoder],
		flow=Flow[run.config.flow]).to(device)
	num_parameters = sum(p.numel() for p in traj_flow.parameters() if p.requires_grad)
	wandb.log({'parameters': num_parameters})

	train_start_time = time.time()

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
		
	train_end_time = time.time()
	train_runtime = train_end_time - train_start_time
	wandb.log({'train runtime': train_runtime})

	traj_flow.eval()
	inputs, features = next(iter(ind.observation_site1.test_loader))
	input = inputs[:, :100, ...].to(device)
	feature = features[:, :100, ...].to(device)
	inference_start_time = time.time()
	traj_flow.sample(input, feature, 100)
	inference_end_time = time.time()
	inference_runtime = inference_end_time - inference_start_time
	if verbose:
		print(inference_runtime)
	wandb.log({'inference runtime': inference_runtime})
		
	for loss in total_loss:
		wandb.log({'loss': loss})
			
	if should_serialize:
		model_name = 'traj_flow_CDE_CNF_0_190596211.pt'
		#model_name = f'traj_flow_{run.config.encoder}_{run.config.flow}_{run.config.masked_data_ratio}_{run.config.seed}.pt'
		if should_train:
			torch.save(traj_flow.state_dict(), model_name)
		traj_flow.load_state_dict(torch.load(model_name))

	if should_evaluate:
		rmse, crps, nll = evaluate(
			observation_site=ind.observation_site1,
			model=traj_flow,
			num_samples=1000,
			device=device)
		
		if verbose:
			print(f'rmse: {rmse}')
			print(f'crps: {crps}')
			print(f'nll: {nll}')
		wandb.log({'rmse': rmse, 'crps': crps, 'nll': nll})

	if should_visualize:
		visualize(
			observation_site=ind.observation_site1,
			model=traj_flow,
			num_samples=10,
			steps=10,#1000,
			prob_threshold=0.001,
			output_dir='visualization',
			simple=simple_visualization,
			device=device)
