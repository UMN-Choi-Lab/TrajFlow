"""PPT (single-agent variant, Stage I only) under leave-one-recording-out
cross-validation on inD.

DESIGN NOTES:
- We use PPT's Stage-I architecture only: linear-encoder + GPT (causal
  self-attention) + linear-decoder. PPT's Stage II/III layers
  (predictor_Des, traj_decoder_9, traj_decoder_20) are tied to
  ETH/UCY's 12-step horizon decomposed into a 9-th waypoint plus a
  20-th destination, and do not generalize to inD's 100-step horizon
  without rewriting the model. We therefore evaluate PPT-Stage-I as a
  representative deterministic transformer baseline on inD-LORO and
  acknowledge in the manuscript that the full multi-stage PPT
  pipeline cannot be fairly transferred to this regime.
- Single-agent (no social attention): each inD trajectory sample is
  treated as an independent batch element; we do not group co-existing
  vehicles into scenes. This matches the data pipeline used by all
  other configurations in the inD experiment.
- Training uses teacher forcing: feed past (100) + GT future (99
  shifted) and supervise the model's prediction at every future step
  with MSE loss against GT future. At inference we autoregressively
  generate 100 future steps from the past, which produces a single
  deterministic trajectory per sample.
"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the local PPT code importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "PPT"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PPT.models.model import GPT
from inference_experiments import build_loro_ind

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = Path("../runs")


def build_config(past_len=100, future_len=100, n_embd=128, n_head=4, n_layer=3):
    return SimpleNamespace(
        vocab_size=2,                 # (x, y)
        block_size=past_len + future_len,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        past_len=past_len,
        future_len=future_len,
        mode="Short_term",
        cuda=True,
    )


class PPTStage1(nn.Module):
    """PPT Stage-I architecture instantiated for inD: 2-D xy → MLP-embed → GPT
    (causal) → MLP-decode → 2-D xy. Trained autoregressively."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.vocab_size, config.n_embd)
        self.gpt = GPT(config)
        self.decoder = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x):
        h = self.encoder(x)
        h = self.gpt(h, mask_type="causal")
        return self.decoder(h)

    @torch.no_grad()
    def autoregressive_predict(self, past, future_len):
        """Greedy single-trajectory inference. past: (B, past_len, 2)."""
        B, past_len, _ = past.shape
        seq = past.clone()
        for _ in range(future_len):
            out = self.forward(seq)
            next_pos = out[:, -1:, :]  # (B, 1, 2)
            seq = torch.cat([seq, next_pos], dim=1)
        return seq[:, past_len:, :]  # (B, future_len, 2)


def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction="mean"))


def crps(y_true, y_pred):
    # Deterministic: y_pred is shape (B, T, 2), repeat to (1, B, T, 2) so the
    # sample-axis collapses to MAE (CRPS for a deterministic predictor reduces
    # to absolute error).
    return torch.mean(torch.abs(y_pred - y_true))


def min_ade(y_true, y_pred):
    return torch.mean(torch.norm(y_pred - y_true, dim=-1))


def min_fde(y_true, y_pred):
    return torch.mean(torch.norm(y_pred[:, -1, :] - y_true[:, -1, :], dim=-1))


def train_one_fold(train_sites, test_sites, num_epochs=50, lr=1e-3, seed=11):
    torch.manual_seed(seed); np.random.seed(seed)
    site = build_loro_ind(train_sites, test_sites)
    config = build_config()
    model = PPTStage1(config).to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {nparams}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    t0 = time.time()
    for epoch in range(num_epochs):
        ep_losses = []
        for inp, _, tgt in site.train_loader:
            inp = inp.to(DEVICE)              # (B, 100, 2)
            tgt = tgt.to(DEVICE)              # (B, 100, 2)
            full = torch.cat([inp, tgt], dim=1)  # (B, 200, 2)
            # Teacher forcing: model sees full[:, :-1], supervises full[:, 1:]
            x_in = full[:, :-1, :]
            y_tgt = full[:, 1:, :]
            pred = model(x_in)
            # Only supervise the future-half predictions (positions 100..199 of full,
            # which are full[:, 100:200, :], predicted from x_in[:, 99:199, :] at
            # output positions 99..198 inclusive).
            past_len = config.past_len
            loss = F.mse_loss(pred[:, past_len - 1:, :], y_tgt[:, past_len - 1:, :])

            optim.zero_grad(); loss.backward(); optim.step()
            ep_losses.append(loss.item())
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"  epoch {epoch:>3d}: loss {sum(ep_losses)/len(ep_losses):.6f}", flush=True)
    train_runtime = time.time() - t0
    print(f"  train runtime: {train_runtime:.1f}s", flush=True)

    # Inference latency
    model.eval()
    with torch.no_grad():
        first_inp, _, _ = next(iter(site.test_loader))
        first_inp = first_inp.to(DEVICE)
        t0 = time.time()
        model.autoregressive_predict(first_inp, future_len=100)
        inference_runtime = time.time() - t0

    # Eval
    rmse_sum = crps_sum = mad_sum = mfd_sum = 0.0
    count = 0
    with torch.no_grad():
        for test_inp, _, test_tgt in site.test_loader:
            test_inp = test_inp.to(DEVICE)
            test_tgt = test_tgt.to(DEVICE)
            pred = model.autoregressive_predict(test_inp, future_len=100)
            # Denormalize back to meters (inD spatial bounds [25,85] x [-65,-10])
            test_tgt_m = torch.tensor(site.denormalize(test_tgt.cpu().numpy())).to(DEVICE)
            pred_m = torch.tensor(site.denormalize(pred.cpu().numpy())).to(DEVICE)

            rmse_sum += rmse(test_tgt_m, pred_m).item()
            crps_sum += crps(test_tgt_m, pred_m).item()
            mad_sum += min_ade(test_tgt_m, pred_m).item()
            mfd_sum += min_fde(test_tgt_m, pred_m).item()
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
    }


def main():
    folds = [
        (["07", "09", "10"], ["08"]),
        (["07", "08", "10"], ["09"]),
        (["07", "08", "09"], ["10"]),
        (["08", "09", "10"], ["07"]),
    ]
    for train_sites, test_sites in folds:
        tag = f"loro_pptstage1_train-{'-'.join(train_sites)}_test-{test_sites[0]}"
        out = RUNS / f"{tag}.json"
        if out.exists():
            print(f"===== SKIP  {tag} =====", flush=True)
            continue
        print(f"===== START {tag} =====", flush=True)
        res = train_one_fold(train_sites, test_sites)
        res.update({"tag": tag, "model": "PPT-Stage1-singleagent",
                    "train_sites": train_sites, "test_sites": test_sites})
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  -> RMSE={res['rmse']:.3f}  CRPS={res['crps']:.3f}  minADE={res['min_ade']:.3f}  minFDE={res['min_fde']:.3f}", flush=True)
        print(f"===== DONE  {tag} =====", flush=True)


if __name__ == "__main__":
    main()
