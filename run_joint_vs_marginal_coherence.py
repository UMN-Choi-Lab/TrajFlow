"""R1#2 deliverable: σ_end and per-step jerk for joint vs marginal CDE-CNF
on the inD 75/25 single-recording split.

Reads the two trained checkpoints (marginal: r1q4_seed11_cdecnf-marg_ind08;
joint: r1q2_cdecnf_joint_ind75_25), draws N=20 samples per scene on the
inD test set (recording 08, 75/25 split), and computes:
  - endpoint dispersion σ_end (per-scene std of the endpoint across N samples,
    in metres after denormalization to inD bounds)
  - mean per-step jerk (third derivative of position, METRES — denormalized
    via site.denormalize so units match the §5.6 sampling-strategy table)
The output JSON is consumed by the §5.1 trade-off discussion.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from datasets.InD import InD
from inference_experiments import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = Path("../runs")
CKPT = RUNS / "checkpoints"


def per_scene_endpoint_std_meters(samples_norm, denorm_fn):
    """samples: (N, T, 2) in normalized coords. Returns std of endpoint (in metres)."""
    end = samples_norm[:, -1, :]                     # (N, 2)
    end_m = denorm_fn(end[None].cpu().numpy())[0]    # (N, 2) in metres
    return float(np.linalg.norm(end_m.std(axis=0)))


def mean_per_step_jerk_meters(samples_norm, denorm_fn):
    """Mean L2 norm of jerk (third diff) in METRES, matching the units used in Table 7.

    samples_norm: (N, T, 2) in normalized inD coordinates. The inD denormalization
    is anisotropic (x-range 60 m, y-range 55 m), so we denormalize before computing
    the third difference rather than applying a single scalar.
    """
    s_norm = samples_norm.cpu().numpy()              # (N, T, 2)
    s_m = denorm_fn(s_norm)                          # (N, T, 2) in metres
    jerk = np.diff(s_m, n=3, axis=1)                 # (N, T-3, 2)
    return float(np.mean(np.linalg.norm(jerk, axis=-1)))


def evaluate(ckpt_path, marginal, num_samples=20, num_test_scenes=200):
    print(f"\n=== {ckpt_path.name} (marginal={marginal}) ===", flush=True)
    ind = InD(root="data", train_ratio=0.75, train_batch_size=64, test_batch_size=1, missing_rate=0)
    site = ind.observation_site1
    model = load_model(str(ckpt_path), encoder="CDE", flow="CNF", marginal=marginal, device=DEVICE)

    end_stds = []; jerks = []
    test_iter = iter(site.test_loader)
    for i in range(num_test_scenes):
        try:
            inp, feat, tgt = next(test_iter)
        except StopIteration:
            break
        inp = inp.to(DEVICE); feat = feat.to(DEVICE)
        with torch.no_grad():
            _, samples, _ = model.sample(inp, feat, tgt.shape[1], num_samples)
        s = samples
        if s.ndim == 4:
            s = s[:, 0]
        end_stds.append(per_scene_endpoint_std_meters(s, site.denormalize))
        jerks.append(mean_per_step_jerk_meters(s, site.denormalize))
    return {
        "n_scenes": int(len(end_stds)),
        "endpoint_std_m_mean": float(np.mean(end_stds)),
        "endpoint_std_m_median": float(np.median(end_stds)),
        "jerk_mean": float(np.mean(jerks)),
        "jerk_median": float(np.median(jerks)),
    }


def main():
    marg_ckpt = CKPT / "r1q4_seed11_cdecnf-marg_ind08.pt"
    joint_ckpt = CKPT / "r1q2_cdecnf_joint_ind75_25.pt"
    assert marg_ckpt.exists(), f"missing {marg_ckpt}"
    assert joint_ckpt.exists(), f"missing {joint_ckpt}"

    marg = evaluate(marg_ckpt, marginal=True)
    joint = evaluate(joint_ckpt, marginal=False)
    out = {
        "tag": "r1q2_joint_vs_marginal_coherence",
        "protocol": "inD 75/25 single-session, recording 08, N=20 samples per scene",
        "marginal": marg,
        "joint": joint,
    }
    with open(RUNS / "r1q2_joint_vs_marginal_coherence.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== FINAL ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
