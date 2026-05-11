"""PPT-Stage-1 sanity check on the inD 75/25 single-recording split.

Goal: reproduce (within reason) the published RMSE = 1.16 m for PPT on
inD's 75/25 split (recording 08, train_ratio=0.75). If the simplified
single-agent Stage-I architecture lands in the same ballpark, we have
confidence the LORO numbers will be a defensible PPT-style baseline.
If it lands far off (e.g. 5.0 or 0.5), Stage-I alone is not enough
and we need to acknowledge that the LORO results are a single-stage
approximation rather than full-pipeline PPT.
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
from run_ppt_loro import PPTStage1, build_config, rmse, crps, min_ade, min_fde

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = Path("../runs")


def main():
    torch.manual_seed(11)
    ind = InD(root="data", train_ratio=0.75, train_batch_size=64, test_batch_size=1, missing_rate=0)
    site = ind.observation_site1

    config = build_config(past_len=100, future_len=100, n_embd=128, n_head=4, n_layer=3)
    model = PPTStage1(config).to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {nparams:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()
    t0 = time.time()
    for epoch in range(50):
        ep_losses = []
        for inp, _, tgt in site.train_loader:
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)
            full = torch.cat([inp, tgt], dim=1)
            x_in = full[:, :-1, :]
            y_tgt = full[:, 1:, :]
            pred = model(x_in)
            past_len = config.past_len
            loss = F.mse_loss(pred[:, past_len - 1:, :], y_tgt[:, past_len - 1:, :])
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            ep_losses.append(loss.item())
        if epoch % 5 == 0 or epoch == 49:
            print(f"  epoch {epoch:>3d}: loss {sum(ep_losses)/len(ep_losses):.6f}", flush=True)
    train_runtime = time.time() - t0

    model.eval()
    rmse_sum = crps_sum = mad_sum = mfd_sum = 0.0
    count = 0
    with torch.no_grad():
        for test_inp, _, test_tgt in site.test_loader:
            test_inp = test_inp.to(DEVICE)
            test_tgt = test_tgt.to(DEVICE)
            pred = model.autoregressive_predict(test_inp, future_len=100)
            test_tgt_m = torch.tensor(site.denormalize(test_tgt.cpu().numpy())).to(DEVICE)
            pred_m = torch.tensor(site.denormalize(pred.cpu().numpy())).to(DEVICE)
            rmse_sum += rmse(test_tgt_m, pred_m).item()
            crps_sum += crps(test_tgt_m, pred_m).item()
            mad_sum += min_ade(test_tgt_m, pred_m).item()
            mfd_sum += min_fde(test_tgt_m, pred_m).item()
            count += 1
    res = {
        "tag": "ppt_75_25_sanity",
        "model": "PPT-Stage1-singleagent",
        "rmse": rmse_sum / count,
        "crps": crps_sum / count,
        "min_ade": mad_sum / count,
        "min_fde": mfd_sum / count,
        "params": int(nparams),
        "train_runtime_s": float(train_runtime),
        "epochs": 50,
        "target_rmse_published_paper": 1.16,
    }
    with open(RUNS / "ppt_75_25_sanity.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nRESULT: RMSE={res['rmse']:.3f}  CRPS={res['crps']:.3f}  minADE={res['min_ade']:.3f}  minFDE={res['min_fde']:.3f}", flush=True)
    print(f"vs published PPT RMSE 1.16 — {'CLOSE' if abs(res['rmse']-1.16)<0.30 else 'FAR'}", flush=True)


if __name__ == "__main__":
    main()
