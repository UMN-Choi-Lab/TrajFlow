"""Inference-only experiments for the TrajFlow AIT revision.

Given a trained CDE-CNF marginal checkpoint on inD, this script produces:

  (R1#3 / R2#4) sampling-strategy ablation:
    naive, top-k (k in {5, 10, 20}), MAP, smoothness-regularized (Viterbi-style)
    reported with minADE, minFDE, endpoint dispersion, mean per-step jerk.

  (R1#2) marginal vs joint diversity/coherence diagnostics:
    endpoint std and mean per-step angular change. (Requires both checkpoints.)

  (R1#5) metric-sensitivity curve:
    evaluate minADE, minFDE, RMSE, CRPS at N in {5, 10, 20, 50, 100, 1000}
    on inD first (since ETH/UCY needs its own checkpoint).

Each sub-experiment appends a JSON to --out_dir.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F

from datasets.Dataset import Dataset
from datasets.InD import InD, InDDataset, InDObservationSite, normalize as ind_normalize
from model.TrajFlow import CausalEnocder, Flow, TrajFlow


def load_model(checkpoint, encoder="CDE", flow="CNF", marginal=True, device="cuda"):
    model = TrajFlow(
        seq_len=100,
        input_dim=2,
        feature_dim=5,
        embedding_dim=128,
        hidden_dim=512,
        causal_encoder=CausalEnocder[encoder],
        flow=Flow[flow],
        marginal=marginal,
        norm_rotation=False,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_default_ind(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ind = InD(root="data", train_ratio=0.75, train_batch_size=64, test_batch_size=1, missing_rate=0)
    return ind.observation_site1


def build_loro_ind(train_sites, test_sites):
    from torch.utils.data import DataLoader

    spatial_boundaries = np.array([[25, 85], [-65, -10]])
    feature_boundaries = np.array([[0, 360], [-10, 10], [-10, 10], [-5, 5], [-5, 5]])
    ind = InD(root="data", train_ratio=1.0, train_batch_size=64, test_batch_size=1)

    def _collect(site_list):
        ins, feats_all = [], []
        for site in site_list:
            inp, feats, _ = ind._parse(site)
            inp = ind_normalize(inp, spatial_boundaries)
            feats = ind_normalize(feats, feature_boundaries)
            ins.append(torch.FloatTensor(inp))
            feats_all.append(torch.FloatTensor(feats))
        inp = torch.cat(ins, dim=0)
        feats = torch.cat(feats_all, dim=0)
        feats = ind._append_time(feats)
        ind._mask(inp, feats)
        return InDDataset(inp, feats)

    train_set = _collect(train_sites)
    test_set = _collect(test_sites)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    background = os.path.join("data", f"{test_sites[0]}_background.png")
    return InDObservationSite(background, 0, spatial_boundaries, train_loader, test_loader)


# -------------------- metrics --------------------

def _rmse(y_true, y_pred):  # y_pred: (N, seq, 2); y_true: (1, seq, 2)
    return torch.sqrt(F.mse_loss(y_true.expand_as(y_pred), y_pred, reduction="mean"))


def _crps(y_true, y_pred):
    n = y_pred.shape[0]
    absolute_error = torch.mean(torch.abs(y_pred - y_true), dim=0)
    if n == 1:
        return torch.mean(absolute_error)
    y_sorted, _ = torch.sort(y_pred, dim=0)
    empirical_cdf = torch.arange(n, device=y_pred.device).view(-1, 1, 1) / n
    b0 = torch.mean(y_sorted, dim=0)
    b1 = torch.mean(y_sorted * empirical_cdf, dim=0)
    return torch.mean(absolute_error + b0 - 2 * b1)


def _min_ade(y_true, y_pred):
    d = torch.norm(y_pred - y_true.expand_as(y_pred), dim=-1)
    return torch.mean(torch.min(d, dim=0).values)


def _min_fde(y_true, y_pred):
    d = torch.norm(y_pred[:, -1, :] - y_true[:, -1, :], dim=-1)
    return d.min()


def _endpoint_std(samples):  # (N, seq, 2) -> scalar meters
    ends = samples[:, -1, :]
    return torch.mean(torch.std(ends, dim=0)).item()


def _mean_step_jerk(samples):  # mean abs change in acceleration
    vel = samples[:, 1:, :] - samples[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    jerk = acc[:, 1:, :] - acc[:, :-1, :]
    return torch.mean(torch.norm(jerk, dim=-1)).item()


def _mean_step_angle(samples):  # mean abs turning angle per step (deg)
    vel = samples[:, 1:, :] - samples[:, :-1, :]
    v1 = vel[:, :-1, :]
    v2 = vel[:, 1:, :]
    cos = (v1 * v2).sum(-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8)
    cos = cos.clamp(-1, 1)
    ang = torch.acos(cos) * 180.0 / math.pi
    return torch.mean(ang).item()


# -------------------- sampling strategies --------------------

def _sample_candidates(model, obs, feat, futures, n_candidates, device):
    """Return (n_candidates, futures, 2) candidate positions AND their per-step marginal log-probs.

    We draw n_candidates independent samples and also get back the flow's log-prob of each sample
    at each step, which we use to score candidates for top-k / smoothness-regularized decoding.
    """
    with torch.no_grad():
        _, samples, delta_logpz = model.sample(obs, feat, futures, n_candidates)
        # log-prob at each step requires log_prob, but TrajFlow only returns aggregated delta_logpz
        # for the marginal formulation we approximate step-wise log-prob from delta_logpz shape
        # (n_candidates, futures) — this is actually already per-step in the marginal setup.
    return samples, delta_logpz


def naive_sample(model, obs, feat, futures, num_samples, device):
    with torch.no_grad():
        _, samples, _ = model.sample(obs, feat, futures, num_samples)
    return samples


def top_k_strategy(model, obs, feat, futures, num_samples, k, device):
    """Draw k*num_samples candidates, then for each of num_samples output trajectories pick
    the step-wise highest-density candidate among its k options at each step."""
    # Shortcut: draw num_samples * k samples, then at each step keep argmax of the density proxy
    # (we use delta_logpz per step as the density proxy since the flow is marginal).
    total = num_samples * k
    with torch.no_grad():
        _, samples, logp = model.sample(obs, feat, futures, total)
    # samples: (total, futures, 2); logp: (total, futures)
    samples = samples.view(num_samples, k, futures, 2)
    logp = logp.view(num_samples, k, futures)
    # At each step, select the candidate in its group of k with highest log-prob
    idx = logp.argmax(dim=1)  # (num_samples, futures)
    picked = torch.gather(
        samples,
        dim=1,
        index=idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, futures, 2),
    ).squeeze(1)
    return picked  # (num_samples, futures, 2)


def map_path(model, obs, feat, futures, num_samples, device, k=100):
    """Approximate argmax_u p(u|O,F) at each step by drawing k samples and taking the one with
    the largest per-step density (log-prob). Repeat num_samples times with different draws to
    give a rough distribution of the MAP estimator (should nearly collapse to a single path)."""
    with torch.no_grad():
        _, samples, logp = model.sample(obs, feat, futures, num_samples * k)
    samples = samples.view(num_samples, k, futures, 2)
    logp = logp.view(num_samples, k, futures)
    idx = logp.argmax(dim=1)
    picked = torch.gather(
        samples,
        dim=1,
        index=idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, futures, 2),
    ).squeeze(1)
    return picked


def smoothness_viterbi(model, obs, feat, futures, num_samples, device, k=20, lam=10.0):
    """For each output trajectory, draw k candidates per step and solve a DP that maximizes
    sum_t logp_t - lam * ||u_t - u_{t-1}||^2."""
    outputs = []
    with torch.no_grad():
        for _ in range(num_samples):
            _, samples, logp = model.sample(obs, feat, futures, k)
            # samples: (k, futures, 2); logp: (k, futures)
            samples = samples.squeeze(0) if samples.dim() == 4 else samples
            # we want per-step candidate grids
            # at step t, the k candidates are samples[:, t, :]
            # DP: best_score[t, i] = max over j of best_score[t-1, j] + logp[i, t] - lam * ||samples[i,t] - samples[j,t-1]||^2
            T = futures
            scores = torch.full((T, k), float("-inf"), device=samples.device)
            prev = torch.zeros((T, k), dtype=torch.long, device=samples.device)
            scores[0] = logp[:, 0]
            for t in range(1, T):
                d = torch.cdist(samples[:, t, :], samples[:, t - 1, :], p=2) ** 2  # (k, k)
                cand = scores[t - 1].unsqueeze(0) - lam * d + logp[:, t].unsqueeze(1)
                best_val, best_idx = cand.max(dim=1)
                scores[t] = best_val
                prev[t] = best_idx
            # backtrack
            last = scores[-1].argmax().item()
            path = [last]
            for t in range(T - 1, 0, -1):
                last = prev[t, last].item()
                path.append(last)
            path = list(reversed(path))
            idxs = torch.tensor(path, device=samples.device)
            picked = samples[idxs, torch.arange(T, device=samples.device)]  # (T, 2)
            outputs.append(picked)
    return torch.stack(outputs, dim=0)  # (num_samples, futures, 2)


# -------------------- drivers --------------------

def run_sampling_ablation(checkpoint, out_dir, num_test=50, num_samples=20, device="cuda"):
    model = load_model(checkpoint, device=device)
    site = build_default_ind(seed=0)
    results = {s: {"min_ade": [], "min_fde": [], "endpoint_std": [], "jerk": []} for s in [
        "naive", "top5", "top10", "top20", "map", "viterbi"
    ]}
    boundaries = site.boundaries
    test_iter = iter(site.test_loader)
    for i in range(num_test):
        try:
            inp, feat, tgt = next(test_iter)
        except StopIteration:
            break
        inp, feat, tgt = inp.to(device), feat.to(device), tgt.to(device)
        futures = tgt.shape[1]
        tgt_unn = torch.tensor(site.denormalize(tgt.cpu().numpy())).to(device)
        for name, fn in [
            ("naive", lambda: naive_sample(model, inp, feat, futures, num_samples, device)),
            ("top5", lambda: top_k_strategy(model, inp, feat, futures, num_samples, 5, device)),
            ("top10", lambda: top_k_strategy(model, inp, feat, futures, num_samples, 10, device)),
            ("top20", lambda: top_k_strategy(model, inp, feat, futures, num_samples, 20, device)),
            ("map", lambda: map_path(model, inp, feat, futures, num_samples, device, k=50)),
            ("viterbi", lambda: smoothness_viterbi(model, inp, feat, futures, num_samples, device, k=20, lam=10.0)),
        ]:
            try:
                samp = fn()
                samp_unn = torch.tensor(site.denormalize(samp.cpu().numpy())).to(device)
                results[name]["min_ade"].append(_min_ade(tgt_unn, samp_unn).item())
                results[name]["min_fde"].append(_min_fde(tgt_unn, samp_unn).item())
                results[name]["endpoint_std"].append(_endpoint_std(samp_unn))
                results[name]["jerk"].append(_mean_step_jerk(samp_unn))
            except Exception as e:
                print(f"[warn] strategy {name} at instance {i} failed: {e}", flush=True)
    agg = {name: {k: float(np.mean(v)) if v else float("nan") for k, v in d.items()}
           for name, d in results.items()}
    (Path(out_dir) / "sampling_ablation.json").write_text(json.dumps(agg, indent=2))
    print("[result] sampling ablation:", json.dumps(agg, indent=2), flush=True)


def run_marginal_vs_joint(checkpoint_marginal, checkpoint_joint, out_dir, num_test=50, num_samples=100, device="cuda"):
    site = build_default_ind(seed=0)
    summaries = {}
    for label, ckpt, marg in [("marginal", checkpoint_marginal, True), ("joint", checkpoint_joint, False)]:
        if not Path(ckpt).exists():
            print(f"[warn] {label} checkpoint not found: {ckpt}", flush=True)
            continue
        model = load_model(ckpt, marginal=marg, device=device)
        stats = {"endpoint_std": [], "angle_deg": [], "jerk": []}
        test_iter = iter(site.test_loader)
        for _ in range(num_test):
            try:
                inp, feat, tgt = next(test_iter)
            except StopIteration:
                break
            inp, feat = inp.to(device), feat.to(device)
            with torch.no_grad():
                _, samples, _ = model.sample(inp, feat, tgt.shape[1], num_samples)
            samp_unn = torch.tensor(site.denormalize(samples.cpu().numpy())).to(device)
            stats["endpoint_std"].append(_endpoint_std(samp_unn))
            stats["angle_deg"].append(_mean_step_angle(samp_unn))
            stats["jerk"].append(_mean_step_jerk(samp_unn))
        summaries[label] = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    (Path(out_dir) / "marginal_vs_joint.json").write_text(json.dumps(summaries, indent=2))
    print("[result] marginal vs joint:", json.dumps(summaries, indent=2), flush=True)


def run_metric_sensitivity(checkpoint, out_dir, Ns=(5, 10, 20, 50, 100, 1000), num_test=30, device="cuda"):
    model = load_model(checkpoint, device=device)
    site = build_default_ind(seed=0)
    curve = {}
    for N in Ns:
        rmse_vals, crps_vals, mad, mfd = [], [], [], []
        test_iter = iter(site.test_loader)
        for _ in range(num_test):
            try:
                inp, feat, tgt = next(test_iter)
            except StopIteration:
                break
            inp, feat, tgt = inp.to(device), feat.to(device), tgt.to(device)
            futures = tgt.shape[1]
            with torch.no_grad():
                _, samples, _ = model.sample(inp, feat, futures, N)
            tgt_unn = torch.tensor(site.denormalize(tgt.cpu().numpy())).to(device)
            samp_unn = torch.tensor(site.denormalize(samples.cpu().numpy())).to(device)
            rmse_vals.append(_rmse(tgt_unn, samp_unn).item())
            crps_vals.append(_crps(tgt_unn, samp_unn).item())
            mad.append(_min_ade(tgt_unn, samp_unn).item())
            mfd.append(_min_fde(tgt_unn, samp_unn).item())
        curve[str(N)] = {
            "rmse": float(np.mean(rmse_vals)),
            "crps": float(np.mean(crps_vals)),
            "min_ade": float(np.mean(mad)),
            "min_fde": float(np.mean(mfd)),
        }
    (Path(out_dir) / "metric_sensitivity.json").write_text(json.dumps(curve, indent=2))
    print("[result] metric sensitivity:", json.dumps(curve, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=["sampling", "marginal_vs_joint", "metric_sensitivity"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_joint", default="", help="for marginal_vs_joint only")
    parser.add_argument("--out_dir", default="../runs")
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if args.experiment == "sampling":
        run_sampling_ablation(args.checkpoint, args.out_dir, num_test=args.num_test,
                              num_samples=args.num_samples, device=device)
    elif args.experiment == "marginal_vs_joint":
        run_marginal_vs_joint(args.checkpoint, args.checkpoint_joint, args.out_dir,
                              num_test=args.num_test, num_samples=args.num_samples, device=device)
    elif args.experiment == "metric_sensitivity":
        run_metric_sensitivity(args.checkpoint, args.out_dir, num_test=args.num_test, device=device)
    print(f"[time] {args.experiment} took {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
