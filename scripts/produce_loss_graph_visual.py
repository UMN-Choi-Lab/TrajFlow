import wandb
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

api = wandb.Api()

sweep_id = '05upvutu'
sweep = api.sweep(f'mitchkos21-university-of-minnesota/trajflow_dry_run/{sweep_id}')

num_epochs = 25

loss_dictionary = defaultdict(lambda: [0] * num_epochs)
loss_counts = defaultdict(int)

for run in sweep.runs:
    config = run.config
    encoder = config['encoder']
    flow = config['flow']
    masked_data_ratio = config['masked_data_ratio']
    key = f'{encoder}-{flow}'

    if masked_data_ratio == 0 and 'loss' in run.history():
        losses = run.history()['loss'].tolist()
        losses = [loss for loss in losses if not (isinstance(loss, float) and np.isnan(loss))]
        loss_dictionary[key] = [loss_dictionary[key][i] + losses[i] for i in range(num_epochs)]
        loss_counts[key] += 1

for key in loss_dictionary.keys():
    loss_dictionary[key] = [loss_dictionary[key][i] / loss_counts[key] for i in range(num_epochs)]

plt.figure(figsize=(10, 6))
#for label, values in loss_dictionary.items():
for key in ['GRU-DNF', 'GRU-CNF', 'CDE-DNF', 'CDE-CNF']:
    values = loss_dictionary[key]
    plt.plot(range(1, num_epochs + 1), values, label=key)
plt.xlabel('Epoch')
plt.ylabel('NLL')
plt.legend()
plt.grid()
plt.savefig('model_loss.pdf', format='pdf')
plt.show()