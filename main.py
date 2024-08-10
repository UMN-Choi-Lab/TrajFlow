import multiprocessing
import random
import wandb
import torch
from InD import InD
from model.TrajFlow import TrajFlow,  CausalEnocder, Flow
from train import train
from evaluate import evaluate
from visualize import visualize

sweep_config = {
	'method': 'grid',
	'metric': {
		'name': 'crps',
		'goal': 'minimize'   
	},
	'parameters': {
		'encoder': {
			'values': ['GRU', 'CDE'],
		},
		'flow': {
			'values': ['DNF', 'CNF']
		},
		'masked_data_ratio': {
			'values': [0, 0.3, 0.5, 0.7]
		},
		'seed': {
			'values': [random.randint(0, 2**32 - 1) for _ in range(2)]
		}
	}
}

def sweep_agent(gpu):
	def agent(config=None):
		should_train = False
		should_serialize = False
		should_evaluate = False
		should_visualize = False

		with wandb.init(config):
			config = wandb.config

			torch.manual_seed(config.seed)

			ind_train = InD(
				root="data",
				train_ratio=0.9999, 
				train_batch_size=64, 
				test_batch_size=1,
				missing_rate=config.masked_data_ratio)

			ind_test = InD(
				root="data",
				train_ratio=0.0001, 
				train_batch_size=64, 
				test_batch_size=1,
				missing_rate=config.masked_data_ratio)

			print(gpu)
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
			traj_flow = TrajFlow(
				seq_len=100, 
				input_dim=2, 
				feature_dim=5, 
				embedding_dim=128,
				hidden_dims=(512,512,512,512),
				causal_encoder=CausalEnocder[config.encoder],
				flow=Flow[config.flow]).to(device)

			if should_train:
				train(
					observation_site=ind_train.observation_site7,
					model=traj_flow,
					epochs=25,#100,
					lr=1e-3,
					weight_decay=0,#1e-5,
					gamma=0.999,
					verbose=False,
					device=device)
			
			if should_serialize:
				if should_train:
					torch.save(traj_flow.state_dict(), 'traj_flow.pt')
				traj_flow.load_state_dict(torch.load('traj_flow.pt'))

			if should_evaluate:
				rmse, crps = evaluate(
					observation_site=ind_test.observation_site8,
					model=traj_flow,
					num_samples=1,
					device=device)
				print(f'RMSE: {rmse}')
				print(f'crps: {crps}')
				wandb.log({'rmse': rmse, 'crps': crps}) 

			if should_visualize:
				visualize(
					observation_site=ind_test.observation_site8,
					model=traj_flow,
					num_samples=10,
					steps=100,
					prob_threshold=0.001,
					output_dir='visualization',
					device=device)
	return agent

def dispatch_agent(sweep_id, count, gpu):
	print(f'agent:{gpu}')
	wandb.agent(sweep_id, sweep_agent(gpu), count=count)

total_encoders = len(sweep_config['parameters']['encoder']['values'])
total_flows = len(sweep_config['parameters']['flow']['values'])
total_masked_data_ratios = len(sweep_config['parameters']['masked_data_ratio']['values'])
total_seeds = len(sweep_config['parameters']['seed']['values'])
total_experiments = total_encoders * total_flows * total_masked_data_ratios * total_seeds

gpus = 4
agent_workload = total_experiments / gpus

if __name__ == '__main__':
	wandb.login()
	sweep_id = wandb.sweep(sweep_config, project="traj-flow-test")

	processes = []
	for gpu in range(gpus):
		p = multiprocessing.Process(target=dispatch_agent, args=(sweep_id, agent_workload, gpu,))
		processes.append(p)
		p.start()
	[p.join for p in processes]
