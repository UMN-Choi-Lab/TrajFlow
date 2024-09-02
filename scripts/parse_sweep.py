import wandb
from collections import defaultdict

api = wandb.Api()

sweep_id = 'akjlyrid'
sweep = api.sweep(f'mitchkos21-university-of-minnesota/trajflow_dry_run/{sweep_id}')

config_counts = defaultdict(int)
nll_results = defaultdict(int)
crps_results = defaultdict(int)
rmse_results = defaultdict(int)
train_runtime_results = defaultdict(int)
inference_runtime_results = defaultdict(int)
parameters_results = defaultdict(int)

for run in sweep.runs:
    config = run.config
    encoder = config['encoder']
    flow = config['flow']
    masked_data_ratio = config['masked_data_ratio']
    key = f'{encoder}-{flow}-{masked_data_ratio}'

    if encoder != "CDE" or flow != "CNF":
        summary = run.summary
        nll_result = summary['nll']
        crps_result = summary['crps']
        rmse_result = summary['rmse']
        train_runtime_result = summary['train runtime']
        inference_runtime_result = summary['inference runtime']
        parameters_result = summary['parameters']

        config_counts[key] += 1
        nll_results[key] += nll_result
        crps_results[key] += crps_result
        rmse_results[key] += rmse_result
        train_runtime_results[key] += train_runtime_result
        inference_runtime_results[key] += inference_runtime_result
        parameters_results[key] += parameters_result

for key in config_counts.keys():
    num_runs = config_counts[key]
    nll = nll_results[key] / num_runs
    crps = crps_results[key] / num_runs
    rmse = rmse_results[key] / num_runs
    train_runtime = train_runtime_results[key] / num_runs
    inference_runtime = inference_runtime_results[key] / num_runs
    parameters = parameters_results[key] / num_runs

    print(key)
    print(f'nll: {nll}')
    print(f'crps: {crps}')
    print(f'rmse: {rmse}')
    print(f'train runtime: {train_runtime}')
    print(f'inference runtime: {inference_runtime}')
    print(f'parameters: {parameters}')
    print('')
