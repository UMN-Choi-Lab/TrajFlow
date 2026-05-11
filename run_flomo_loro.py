"""FloMo under leave-one-recording-out cross-validation on inD.

For each of the 4 LORO folds, train FloMo for 25 epochs on three of the
four recordings {07, 08, 09, 10} and evaluate on the held-out one.
Saves per-fold result JSONs to runs/loro_flomo_train-XX-YY-ZZ_test-WW.json
in the same format as the existing CDE-CNF and GRU-DNF LORO results.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from inference_experiments import build_loro_ind
from model.FloMo import FloMo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = Path("../runs")


# Same metric definitions as train_and_evaluate_flomo_standard.py
def rmse(y_true, y_pred):
    mse = F.mse_loss(y_true.expand_as(y_pred), y_pred, reduction="mean")
    return torch.sqrt(mse)


def crps(y_true, y_pred):
    num_samples = y_pred.shape[0]
    absolute_error = torch.mean(torch.abs(y_pred - y_true), dim=0)
    if num_samples == 1:
        return torch.mean(absolute_error)
    y_pred, _ = torch.sort(y_pred, dim=0)
    empirical_cdf = torch.arange(num_samples, device=y_pred.device).view(-1, 1, 1) / num_samples
    b0 = torch.mean(y_pred, dim=0)
    b1 = torch.mean(y_pred * empirical_cdf, dim=0)
    return torch.mean(absolute_error + b0 - 2 * b1)


def min_ade(y_true, y_pred):
    distances = torch.norm(y_pred - y_true.expand_as(y_pred), dim=-1)
    return torch.mean(torch.min(distances, dim=0).values)


def min_fde(y_true, y_pred):
    fde = torch.norm(y_pred[:, -1, :] - y_true[:, -1, :], dim=-1)
    return fde.min()


def train_one_fold(train_sites, test_sites, num_epochs=25, num_eval_samples=100, seed=11):
    torch.manual_seed(seed)
    site = build_loro_ind(train_sites, test_sites)

    flomo = FloMo(
        hist_size=100, pred_steps=100, alpha=1, beta=0, gamma=0,
        num_in=2, num_feat=0, norm_rotation=True,
    ).to(DEVICE)
    nparams = sum(p.numel() for p in flomo.parameters() if p.requires_grad)
    print(f"  parameters: {nparams}")

    optim = torch.optim.Adam(flomo.parameters(), lr=5e-4, weight_decay=0)

    flomo.train()
    t0 = time.time()
    for epoch in range(num_epochs):
        ep_losses = []
        skipped = 0
        for input, _, target in site.train_loader:
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            input = torch.FloatTensor(site.denormalize(input.cpu().numpy())).to(DEVICE)
            target = torch.FloatTensor(site.denormalize(target.cpu().numpy())).to(DEVICE)

            full = torch.cat([input, target], dim=1)
            tmean = full.mean(dim=1, keepdim=True)
            input = input - tmean
            target = target - tmean

            try:
                log_prob = flomo.log_prob(target, input)
                loss = -torch.mean(log_prob)
                optim.zero_grad()
                loss.backward()
                # Gradient clipping protects against the RQS spline discriminant
                # going negative on rare unstable batches at long horizons.
                torch.nn.utils.clip_grad_norm_(flomo.parameters(), max_norm=1.0)
                optim.step()
                ep_losses.append(loss.item())
            except (AssertionError, RuntimeError) as e:
                skipped += 1
                optim.zero_grad()
                continue
        if skipped > 0:
            print(f"  epoch {epoch:>3d}: skipped {skipped} unstable batch(es)", flush=True)
        print(f"  epoch {epoch:>3d}: loss {sum(ep_losses)/len(ep_losses):.4f}", flush=True)
    train_runtime = time.time() - t0
    print(f"  train runtime: {train_runtime:.1f} s", flush=True)

    # Inference latency on a single test sample
    flomo.eval()
    with torch.no_grad():
        first_inp, _, _ = next(iter(site.test_loader))
        first_inp = first_inp.to(DEVICE)
        first_inp = torch.FloatTensor(site.denormalize(first_inp.cpu().numpy())).to(DEVICE)
        first_inp = first_inp - first_inp.mean(dim=1, keepdim=True)
        t0 = time.time()
        flomo.sample(num_eval_samples, first_inp)
        inference_runtime = time.time() - t0

    # Eval over the full test set
    rmse_sum = crps_sum = mad_sum = mfd_sum = 0.0
    count = 0
    with torch.no_grad():
        for test_input, _, test_target in site.test_loader:
            test_input = test_input.to(DEVICE)
            test_target = test_target.to(DEVICE)
            test_input = torch.FloatTensor(site.denormalize(test_input.cpu().numpy())).to(DEVICE)
            test_target = torch.FloatTensor(site.denormalize(test_target.cpu().numpy())).to(DEVICE)
            full = torch.cat([test_input, test_target], dim=1)
            tmean = full.mean(dim=1, keepdim=True)
            test_input = test_input - tmean
            test_target = test_target - tmean

            samples, _ = flomo.sample(num_eval_samples, test_input)
            samples = samples[:, :test_target.shape[1], :]

            rmse_sum += rmse(test_target, samples).item()
            crps_sum += crps(test_target, samples).item()
            mad_sum += min_ade(test_target, samples).item()
            mfd_sum += min_fde(test_target, samples).item()
            count += 1
    return {
        "rmse": rmse_sum / count,
        "crps": crps_sum / count,
        "min_ade": mad_sum / count,
        "min_fde": mfd_sum / count,
        "params": int(nparams),
        "train_runtime_s": float(train_runtime),
        "inference_runtime_s": float(inference_runtime),
        "epochs": int(num_epochs),
        "eval_samples": int(num_eval_samples),
    }


def main():
    folds = [
        (["07", "09", "10"], ["08"]),
        (["07", "08", "10"], ["09"]),
        (["07", "08", "09"], ["10"]),
        (["08", "09", "10"], ["07"]),
    ]
    for train_sites, test_sites in folds:
        tag = f"loro_flomo_train-{'-'.join(train_sites)}_test-{test_sites[0]}"
        out = RUNS / f"{tag}.json"
        if out.exists():
            print(f"===== SKIP  {tag} (result exists) =====", flush=True)
            continue
        print(f"===== START {tag} =====", flush=True)
        res = train_one_fold(train_sites, test_sites)
        res.update({"tag": tag, "train_sites": train_sites, "test_sites": test_sites,
                    "model": "FloMo", "marginal": False, "dataset": "InD-LORO"})
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  -> RMSE={res['rmse']:.3f}  CRPS={res['crps']:.3f}  minADE={res['min_ade']:.3f}  minFDE={res['min_fde']:.3f}", flush=True)
        print(f"===== DONE  {tag} =====", flush=True)


if __name__ == "__main__":
    main()
