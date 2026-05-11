# TrajFlow
Official repository of the [TrajFlow](https://arxiv.org/abs/2501.14266) paper.

![TrajFlow Inference](assets/inference.gif)

# Installation

The code in this repository was implemented against python 3.9. Use the following commands to install the dependecy packackes.

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip install torchdiffeq wandb numpy pandas matplotlib
```

This repository contains a Dockerfile that can be used to setup a clean environment.

# Training

To train a model from scratch you need to edit main.py. In particular you need to ensure that the should_train variable is set to true. should_train is located at the top of the file. There are four possible configuration of the TrajFlow architecture. GRU-DNF, GRU-CNF, CDE-DNF, and CDE-CNF. GRU/CDE refer to the encoder architecture. GRU is a discrete recurrent neural network while CDE is a continous neural differential equation. Likewise DNF/CNF refers to the normalizing flow architecture where DNF is discrete and CNF is continous. You can configure the encoder and flow architecture by editing the default run config in main.py to the appropriate values. In order to save your model weights should_serialize must be set to true.

# Inference

You can evaluate the model on the validation split by setting should_evaluate to true. You can also visualize estimated densities of samples in the validation split by setting should_visualize to true. In order to visualize ffmpeg must be installed and located in the root directory.



# Revision experiments (AIT, 2026)

The following scripts reproduce experiments added in the *AI for Transportation* major-revision round. Each is self-contained and runs against the data already loaded by `datasets/InD.py`.

| Reviewer item | Script | Produces |
|---|---|---|
| R1#2 (joint vs. marginal coherence) | `run_joint_vs_marginal_coherence.py` | $\sigma_{\text{end}}$ and per-step jerk for both formulations on inD 75/25 (Table 8 of the paper) |
| R1#3 / R1#5 (sampling + metric ablations) | `inference_experiments.py` | sampling-strategy table (Table 7) and minADE/minFDE/RMSE/CRPS-vs-N sensitivity (Figure 4) |
| R1#4 (multi-seed + significance) | `run_ethucy_multiseed.sh`, `run_r1q4_multiseed_ind.sh` | 3-seed ETH/UCY sweep (Tables 1, 2) and inD multi-seed runs |
| R1#4 (FloMo baseline retraining) | `run_flomo_75_25_sanity.py`, `run_flomo_loro.py`, `train_and_evaluate_flomo_standard.py` | FloMo numbers under our preprocessing for Table 3 and Table 4 |
| R2#3 (inD LORO cross-validation) | `run_loro.sh`, `run_post_loro_queue.sh` | leave-one-recording-out across $\{07,08,09,10\}$ (Table 4) |
| R1#1 (collision-risk demo) | `collision_demo.py`, `make_supplementary_figures.py` | multi-agent occupancy and concentration figures (Figures 8–10) |

The PPT baseline (Sun et al., 2024) reuses the public reference implementation; see [iSEE-Laboratory/PPT](https://github.com/iSEE-Laboratory/PPT). We did not re-vendor PPT into this repository.

Run-orchestration shell scripts (`run_*.sh`, `run_*chain.sh`) drive sequential CDE-CNF training jobs and trigger downstream evaluations once checkpoints land; outputs land in a `runs/` directory next to the code.
