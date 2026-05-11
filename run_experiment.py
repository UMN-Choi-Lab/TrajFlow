"""Headless experiment runner for the TrajFlow AIT revision.

Embedded as heredoc into vessl YAMLs — so treat it as a self-contained script.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure the current working directory (the cloned TrajFlow repo) wins over
# any globally installed `datasets` package (e.g. HuggingFace datasets).
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from datasets.Dataset import Dataset
from datasets.EthUcy import EthUcy
from datasets.InD import InD, InDDataset, InDObservationSite, normalize
from evaluate import evaluate
from model.TrajFlow import CausalEnocder, Flow, TrajFlow
from train import train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loro_ind_site(root, train_sites, test_sites, train_batch_size=64):
    """Leave-one-recording-out for inD: explicit train/test recording lists."""
    from torch.utils.data import DataLoader

    spatial_boundaries = np.array([[25, 85], [-65, -10]])
    feature_boundaries = np.array([[0, 360], [-10, 10], [-10, 10], [-5, 5], [-5, 5]])
    ind = InD(root=root, train_ratio=1.0, train_batch_size=train_batch_size, test_batch_size=1)

    def _collect(site_list):
        ins, feats_all = [], []
        for site in site_list:
            inp, feats, _ = ind._parse(site)
            inp = normalize(inp, spatial_boundaries)
            feats = normalize(feats, feature_boundaries)
            ins.append(torch.FloatTensor(inp))
            feats_all.append(torch.FloatTensor(feats))
        inp = torch.cat(ins, dim=0)
        feats = torch.cat(feats_all, dim=0)
        feats = ind._append_time(feats)
        ind._mask(inp, feats)
        return InDDataset(inp, feats)

    train_set = _collect(train_sites)
    test_set = _collect(test_sites)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    background = os.path.join(root, f"{test_sites[0]}_background.png")
    return InDObservationSite(background, 0, spatial_boundaries, train_loader, test_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["InD", "EthUcy"], required=True)
    parser.add_argument("--encoder", choices=["GRU", "CDE"], default="CDE")
    parser.add_argument("--flow", choices=["DNF", "CNF"], default="CNF")
    parser.add_argument("--marginal", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--site", default="zara2")
    parser.add_argument("--ind_train_sites", default="")
    parser.add_argument("--ind_test_sites", default="")
    parser.add_argument("--epochs_override", type=int, default=-1)
    parser.add_argument("--eval_samples_override", type=int, default=-1)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--load_only", action="store_true")
    parser.add_argument("--out_dir", default="/root/output")
    parser.add_argument("--tag", default="default")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    causal_encoder = CausalEnocder[args.encoder]
    flow = Flow[args.flow]
    dataset = Dataset[args.dataset]
    marginal = bool(args.marginal)

    if dataset == Dataset.InD:
        seq_len, input_dim, feature_dim = 100, 2, 5
        embedding_dim, hidden_dim = 128, 512
        training_epochs, evaluation_samples = 25, 1000
        norm_rotate = False
        if args.ind_train_sites and args.ind_test_sites:
            train_sites = args.ind_train_sites.split(",")
            test_sites = args.ind_test_sites.split(",")
            observation_site = build_loro_ind_site("data", train_sites, test_sites)
            split_tag = f"train-{'-'.join(train_sites)}_test-{'-'.join(test_sites)}"
        else:
            ind = InD(root="data", train_ratio=0.75, train_batch_size=64, test_batch_size=1, missing_rate=0)
            observation_site = ind.observation_site1
            split_tag = "site08_75-25"
    else:
        seq_len, input_dim, feature_dim = 12, 2, 4
        embedding_dim, hidden_dim = 128, 512
        training_epochs, evaluation_samples = 150, 20
        norm_rotate = True
        ethucy = EthUcy(train_batch_size=128, test_batch_size=1, history=8, futures=12, smin=0.3, smax=1.7)
        site_map = {
            "eth": ethucy.eth_observation_site,
            "hotel": ethucy.hotel_observation_site,
            "univ": ethucy.univ_observation_site,
            "zara1": ethucy.zara1_observation_site,
            "zara2": ethucy.zara2_observation_site,
        }
        observation_site = site_map[args.site]
        split_tag = args.site

    if args.epochs_override > 0:
        training_epochs = args.epochs_override
    if args.eval_samples_override > 0:
        evaluation_samples = args.eval_samples_override

    traj_flow = TrajFlow(
        seq_len=seq_len, input_dim=input_dim, feature_dim=feature_dim,
        embedding_dim=embedding_dim, hidden_dim=hidden_dim,
        causal_encoder=causal_encoder, flow=flow,
        marginal=marginal, norm_rotation=norm_rotate,
    ).to(device)

    num_params = sum(p.numel() for p in traj_flow.parameters() if p.requires_grad)
    print(f"[run] dataset={args.dataset} site={split_tag} seed={args.seed} "
          f"encoder={args.encoder} flow={args.flow} marginal={marginal} "
          f"params={num_params} epochs={training_epochs} eval_samples={evaluation_samples}",
          flush=True)

    train_start = time.time()
    if not args.load_only:
        train(observation_site=observation_site, model=traj_flow,
              epochs=training_epochs, lr=1e-3, weight_decay=0, gamma=0.999,
              verbose=False, device=device)
    train_runtime = time.time() - train_start

    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        if args.load_only:
            traj_flow.load_state_dict(torch.load(args.checkpoint, map_location=device))
        else:
            torch.save(traj_flow.state_dict(), args.checkpoint)

    traj_flow.eval()
    inf_input, inf_feat, _ = next(iter(observation_site.test_loader))
    inf_input = inf_input.to(device)
    inf_feat = inf_feat.to(device)
    inf_start = time.time()
    traj_flow.sample(inf_input, inf_feat, seq_len, 100)
    inf_runtime = time.time() - inf_start

    rmse, crps, min_ade, min_fde, nll = evaluate(
        observation_site=observation_site, model=traj_flow,
        num_samples=evaluation_samples, device=device)

    def _f(x):
        return float(x.item()) if hasattr(x, "item") else float(x)

    result = {
        "tag": args.tag,
        "dataset": args.dataset,
        "site": split_tag,
        "seed": args.seed,
        "encoder": args.encoder,
        "flow": args.flow,
        "marginal": marginal,
        "params": num_params,
        "train_runtime_s": train_runtime,
        "inference_runtime_s": inf_runtime,
        "rmse": _f(rmse),
        "crps": _f(crps),
        "min_ade": _f(min_ade),
        "min_fde": _f(min_fde),
        "nll": _f(nll),
        "eval_samples": evaluation_samples,
        "epochs": training_epochs,
    }
    print("[result]", json.dumps(result, indent=2), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[result] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
