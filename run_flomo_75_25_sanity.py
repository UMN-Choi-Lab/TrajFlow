"""FloMo sanity check on the inD 75/25 single-recording split.

Goal: reproduce the published Table 3 entry of RMSE = 2.39 m for FloMo
on inD (recording 08, train_ratio=0.75, 25 epochs). If our FloMo
implementation lands within ~10% of 2.39, the LORO inflation we
observe (e.g. fold 1 RMSE = 7.72 m) reflects FloMo's actual
cross-recording generalization gap rather than an implementation
issue. If it lands far off 2.39, the LORO numbers may instead be
driven by training-protocol drift (lower learning rate + gradient
clipping that we added for spline stability).

Uses the ORIGINAL hyperparameters from train_and_evaluate_flomo_standard.py:
  - lr = 1e-3 (no clip)
  - 25 epochs
  - 1000 evaluation samples
With a try/except around the spline (the only safety net we keep) so
the run does not crash mid-training on rare unstable batches.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.InD import InD
from model.FloMo import FloMo
from run_flomo_loro import rmse, crps, min_ade, min_fde

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = Path("../runs")


def main(seed=11):
    torch.manual_seed(seed)
    ind = InD(root="data", train_ratio=0.75, train_batch_size=64, test_batch_size=1, missing_rate=0)
    site = ind.observation_site1

    flomo = FloMo(hist_size=100, pred_steps=100, alpha=1, beta=0, gamma=0,
                  num_in=2, num_feat=0, norm_rotation=True).to(DEVICE)
    nparams = sum(p.numel() for p in flomo.parameters() if p.requires_grad)
    print(f"parameters: {nparams:,}", flush=True)

    optim = torch.optim.Adam(flomo.parameters(), lr=1e-3, weight_decay=0)
    flomo.train()
    t0 = time.time()
    for epoch in range(25):
        ep_losses = []; skipped = 0
        for input, _, target in site.train_loader:
            input = input.to(DEVICE); target = target.to(DEVICE)
            input = torch.FloatTensor(site.denormalize(input.cpu().numpy())).to(DEVICE)
            target = torch.FloatTensor(site.denormalize(target.cpu().numpy())).to(DEVICE)
            full = torch.cat([input, target], dim=1)
            tmean = full.mean(dim=1, keepdim=True)
            input = input - tmean; target = target - tmean
            try:
                log_prob = flomo.log_prob(target, input)
                loss = -torch.mean(log_prob)
                optim.zero_grad(); loss.backward(); optim.step()
                ep_losses.append(loss.item())
            except (AssertionError, RuntimeError):
                skipped += 1; optim.zero_grad(); continue
        msg = f"  epoch {epoch:>3d}: loss {sum(ep_losses)/len(ep_losses):.4f}"
        if skipped > 0: msg += f"  (skipped {skipped} batches)"
        print(msg, flush=True)
    train_runtime = time.time() - t0

    flomo.eval()
    rmse_sum = crps_sum = mad_sum = mfd_sum = 0.0
    count = 0
    with torch.no_grad():
        for test_inp, _, test_tgt in site.test_loader:
            test_inp = test_inp.to(DEVICE); test_tgt = test_tgt.to(DEVICE)
            test_inp = torch.FloatTensor(site.denormalize(test_inp.cpu().numpy())).to(DEVICE)
            test_tgt = torch.FloatTensor(site.denormalize(test_tgt.cpu().numpy())).to(DEVICE)
            full = torch.cat([test_inp, test_tgt], dim=1)
            tmean = full.mean(dim=1, keepdim=True)
            test_inp = test_inp - tmean; test_tgt = test_tgt - tmean
            samples, _ = flomo.sample(1000, test_inp)
            samples = samples[:, :test_tgt.shape[1], :]
            rmse_sum += rmse(test_tgt, samples).item()
            crps_sum += crps(test_tgt, samples).item()
            mad_sum += min_ade(test_tgt, samples).item()
            mfd_sum += min_fde(test_tgt, samples).item()
            count += 1
    res = {
        "tag": "flomo_75_25_sanity",
        "model": "FloMo",
        "rmse": rmse_sum / count, "crps": crps_sum / count,
        "min_ade": mad_sum / count, "min_fde": mfd_sum / count,
        "params": int(nparams),
        "train_runtime_s": float(train_runtime),
        "epochs": 25, "eval_samples": 1000, "seed": seed,
        "target_rmse_published_paper": 2.39,
    }
    with open(RUNS / "flomo_75_25_sanity.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nRESULT: RMSE={res['rmse']:.3f}  CRPS={res['crps']:.3f}  minADE={res['min_ade']:.3f}  minFDE={res['min_fde']:.3f}", flush=True)
    print(f"vs published FloMo RMSE 2.39 — {'CLOSE' if abs(res['rmse']-2.39)<0.30 else 'FAR'}", flush=True)


if __name__ == "__main__":
    main()
