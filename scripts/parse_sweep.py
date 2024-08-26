import wandb
from collections import defaultdict

api = wandb.Api()

sweep_id = "mitchkos21-university-of-minnesota/trajflow_dry_run/mh9bdi3a"
sweep = api.sweep(sweep_id)

config_counts = defaultdict(int)
crps_results = defaultdict(int)
rmse_results = defaultdict(int)
runtime_results = defaultdict(int)
parameters_results = defaultdict(int)

for run in sweep.runs:
    config = run.config
    encoder = config['encoder']
    flow = config['flow']
    masked_data_ratio = config['masked_data_ratio']
    key = f'{encoder}-{flow}-{masked_data_ratio}'

    summary = run.summary
    crps_result = summary['crps']
    rmse_result = summary['rmse']
    runtime_result = summary['runtime']
    parameters_result = summary['parameters']

    config_counts[key] += 1
    crps_results[key] += crps_result
    rmse_results[key] += rmse_result
    runtime_results[key] += runtime_result
    parameters_results[key] += parameters_result

for key in config_counts.keys():
    num_runs = config_counts[key]
    crps = crps_results[key] / num_runs
    rmse = rmse_results[key] / num_runs
    runtime = runtime_results[key] / num_runs
    parameters = parameters_results[key] / num_runs

    print(key)
    print(f'crps: {crps}')
    print(f'rmse: {rmse}')
    print(f'runtime: {runtime}')
    print(f'parameters: {parameters}')
    print('')
