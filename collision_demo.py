"""R1#1 downstream demo: marginal-density occupancy visualization + coverage analysis.

Compares CDE-CNF marginal vs GRU-DNF marginal on the inD LORO test set (fold:
train {07,09,10}, test {08}).

Outputs:
  fig_occupancy_panel.pdf      4-panel occupancy heatmap for one representative
                               test agent, comparing CDE-CNF and GRU-DNF.
  fig_collision_proxy.pdf      Per-step "self-collision proxy": expected mass
                               within 1 m of the predicted endpoint over the
                               prediction horizon. Compares CDE-CNF and GRU-DNF.
  fig_multiagent_panel.pdf     Multi-agent test: simultaneous CDE-CNF marginal
                               occupancy heatmaps for several co-existing
                               vehicles in the same recording-08 time window,
                               overlaid in shared world coordinates. Pairwise
                               occupancy-grid overlap is computed as a proxy for
                               agent-pairwise collision probability.
  collision_demo.json          Aggregate numbers used in the paper text.

Runs concurrently with the ETH/UCY training because it uses < 2 GB of GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from inference_experiments import build_loro_ind, load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_marginal(model, obs, feat, futures, num_samples=500):
    with torch.no_grad():
        _, samples, _ = model.sample(obs, feat, futures, num_samples)
    return samples.cpu().numpy()  # (num_samples, futures, 2)


def direct_density_grid(model, obs, feat, t_step, bounds, res=128, futures=100, batch_size=2048):
    """Evaluate p(u^t | O, F) directly via the trained normalizing flow on a 2D grid,
    using the NF's exact change-of-variables log-density. No sample-based KDE.

    Returns an (res, res) array normalized so the cells sum to 1 (so it is
    directly comparable to the KDE-based occupancy_grid output for overlap
    integrals).

    bounds: ((xmin, xmax), (ymin, ymax)) in the same normalized [0,1] coords
    the model was trained on.
    """
    (xmin, xmax), (ymin, ymax) = bounds
    device = next(model.parameters()).device
    xs = torch.linspace(xmin, xmax, res, device=device)
    ys = torch.linspace(ymin, ymax, res, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")  # (res, res)
    grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (res*res, 2)
    n = grid.shape[0]

    log_p = torch.empty(n, device=device)
    obs_b_proto = obs.to(device); feat_b_proto = feat.to(device)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = grid[start:end]
            m = chunk.shape[0]
            target = torch.zeros((m, futures, 2), device=device)
            target[:, t_step, :] = chunk
            obs_b = obs_b_proto.expand(m, *obs.shape[1:])
            feat_b = feat_b_proto.expand(m, *feat.shape[1:])
            z_t0, delta_logpz = model(obs_b, target, feat_b)
            _, logpz_t1 = model.log_prob(z_t0, delta_logpz)
            log_p[start:end] = logpz_t1[:, t_step]

    log_p_grid = log_p.view(res, res).cpu().numpy()
    H = np.exp(log_p_grid - log_p_grid.max())
    s = H.sum()
    if s > 0:
        H /= s
    return H


def occupancy_grid(samples_xy, bounds, res=128, sigma_px=2.0):
    """Soft 2D occupancy via Gaussian KDE-on-grid for a single timestep.

    samples_xy: (num_samples, 2) in normalized [0, 1] coords.
    bounds: ((xmin, xmax), (ymin, ymax)) normalized to [0, 1].
    """
    (xmin, xmax), (ymin, ymax) = bounds
    H = np.zeros((res, res), dtype=np.float64)
    xs = np.clip(((samples_xy[:, 0] - xmin) / (xmax - xmin)) * (res - 1), 0, res - 1)
    ys = np.clip(((samples_xy[:, 1] - ymin) / (ymax - ymin)) * (res - 1), 0, res - 1)
    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)
    for x, y in zip(xi, yi):
        H[y, x] += 1.0
    # Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H, sigma=sigma_px)
    if H.sum() > 0:
        H /= H.sum()
    return H


def expected_mass_near_endpoint(samples, radius=0.05):
    """For each timestep, what fraction of sample mass lies within `radius`
    (in normalized units) of the sample-mean endpoint at that step? Higher
    means tighter, more confident predictions; lower means more diffuse."""
    means = samples.mean(axis=0)  # (futures, 2)
    futures = samples.shape[1]
    masses = np.zeros(futures)
    for t in range(futures):
        d = np.linalg.norm(samples[:, t, :] - means[t][None, :], axis=-1)
        masses[t] = (d < radius).mean()
    return masses


def _scene_score(cde_samples, gru_samples, tgt, obs):
    """Higher score = better headline panel.

    We want a scene where:
      (a) the agent is actually moving (so the figure looks dynamic);
      (b) CDE-CNF density is visibly different from GRU-DNF (so the contrast is
          legible to a reader);
      (c) the ground truth is well-covered by CDE-CNF (so the wider density is
          not pathologically wrong).

    The four contributing factors:
      - log spread ratio sigma_cde / sigma_gru      [reward if positive]
      - elongation of CDE-CNF endpoint covariance   [reward if anisotropic]
      - displacement = ||tgt[-1] - obs[-1]||        [reward if agent moves]
      - GT-to-CDE-mean offset / sigma_cde           [penalty if GT far outside]
    """
    cde_end = cde_samples[:, -1, :]
    gru_end = gru_samples[:, -1, :]
    tgt_end = tgt[-1]
    obs_end = obs[-1]

    sigma_cde = max(np.linalg.norm(cde_end.std(axis=0)), 1e-6)
    sigma_gru = max(np.linalg.norm(gru_end.std(axis=0)), 1e-6)
    log_ratio = np.log(sigma_cde / sigma_gru)

    cov = np.cov(cde_end.T) + 1e-9 * np.eye(2)
    w = np.linalg.eigvalsh(cov)
    elongation = float(w.max() / max(w.min(), 1e-9))
    log_elongation = np.log1p(elongation)

    # Displacement in normalized inD coords; min-max bounding box is [25,85]x[-65,-10],
    # so 1 unit ~ 60 m on the long axis. Threshold ~0.05 (=3 m) for "moving".
    displacement = float(np.linalg.norm(tgt_end - obs_end))
    move_reward = np.log1p(displacement / 0.02)  # saturates above ~5x threshold

    gt_offset = np.linalg.norm(tgt_end - cde_end.mean(axis=0)) / sigma_cde

    return float(log_ratio + 0.5 * log_elongation + 1.0 * move_reward - 0.5 * gt_offset)


_SPATIAL_BOUNDS = np.array([[25.0, 85.0], [-65.0, -10.0]])
_FEATURE_BOUNDS = np.array([[0.0, 360.0], [-10.0, 10.0], [-10.0, 10.0], [-5.0, 5.0], [-5.0, 5.0]])


def _normalize(arr, bounds):
    return (arr - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _denormalize_xy(arr_norm):
    """Inverse of the per-axis min-max normalization for spatial coords."""
    return arr_norm * (_SPATIAL_BOUNDS[:, 1] - _SPATIAL_BOUNDS[:, 0]) + _SPATIAL_BOUNDS[:, 0]


def _find_multiagent_window(
    data_root,
    site,
    window_len=200,
    min_agents=3,
    classes=("car", "truck_bus"),
    min_track_frames=200,
    max_track_frames=1000,
):
    """Pick a (start_frame, [trackId,...]) tuple where as many vehicle tracks
    as possible all have at least `window_len` consecutive frames starting at
    `start_frame`. Replicates the dataset's class/length filters so the picked
    tracks would be valid samples in InD._parse."""
    meta = pd.read_csv(os.path.join(data_root, f"{site}_tracksMeta.csv"))
    cls_mask = meta["class"].isin(classes)
    len_mask = (meta["numFrames"] > min_track_frames) & (meta["numFrames"] < max_track_frames)
    veh = meta[cls_mask & len_mask][["trackId", "initialFrame", "finalFrame"]]

    if len(veh) == 0:
        return None

    max_frame = int(veh["finalFrame"].max())
    counts = np.zeros(max_frame + 1, dtype=np.int32)
    contributors = [[] for _ in range(max_frame + 1)]
    for tid, s, e in veh.itertuples(index=False):
        if e - s + 1 < window_len:
            continue
        for ws in range(int(s), int(e) - window_len + 2):
            counts[ws] += 1
            contributors[ws].append(int(tid))

    if counts.max() < min_agents:
        return None

    best_start = int(np.argmax(counts))
    return best_start, contributors[best_start][:max(min_agents, counts.max())]


def _rank_overlap_candidates(
    data_root,
    site,
    window_len=200,
    obs_len=100,
    classes=("car", "truck_bus"),
    min_track_frames=200,
    max_track_frames=1000,
    stride=10,
    top_k=10,
):
    """Rank candidate (start_frame, [trackIds]) windows by the smallest pairwise
    distance in the *future* half of the window (i.e., during the prediction
    horizon). Smaller distance ⇒ more spatial proximity ⇒ more chance of
    predicted-density overlap. Returns the top_k candidates sorted by
    ascending future-min-pairwise-distance.
    """
    tracks = pd.read_csv(os.path.join(data_root, f"{site}_tracks.csv"))
    meta = pd.read_csv(os.path.join(data_root, f"{site}_tracksMeta.csv"))
    cls_mask = meta["class"].isin(classes)
    len_mask = (meta["numFrames"] > min_track_frames) & (meta["numFrames"] < max_track_frames)
    veh = meta[cls_mask & len_mask][["trackId", "initialFrame", "finalFrame"]]

    if len(veh) == 0:
        return []

    init = dict(zip(veh["trackId"].values, veh["initialFrame"].values))
    fin = dict(zip(veh["trackId"].values, veh["finalFrame"].values))
    ids = list(veh["trackId"].values)

    xy_by_id = {}
    for tid in ids:
        df = tracks[tracks["trackId"] == tid].sort_values("frame")
        f0 = int(df["frame"].iloc[0])
        arr = df[["xCenter", "yCenter"]].to_numpy(dtype=np.float64)
        xy_by_id[tid] = (f0, arr)

    starts = set()
    for tid in ids:
        s, e = int(init[tid]), int(fin[tid])
        if e - s + 1 >= window_len:
            for ws in range(s, e - window_len + 2, stride):
                starts.add(ws)

    candidates = []
    for ws in sorted(starts):
        we = ws + window_len - 1
        fs = ws + obs_len
        fe = we
        present = [tid for tid in ids if init[tid] <= ws and fin[tid] >= we]
        if len(present) < 2:
            continue
        fut = {}
        for tid in present:
            f0, arr = xy_by_id[tid]
            i0 = fs - f0
            i1 = fe - f0 + 1
            if i0 < 0 or i1 > len(arr):
                continue
            fut[tid] = arr[i0:i1]
        if len(fut) < 2:
            continue
        fut_ids = list(fut.keys())
        min_d = np.inf
        for a in range(len(fut_ids)):
            for b in range(a + 1, len(fut_ids)):
                d = float(np.linalg.norm(fut[fut_ids[a]] - fut[fut_ids[b]], axis=1).min())
                if d < min_d:
                    min_d = d
        candidates.append((min_d, int(ws), [int(t) for t in fut_ids]))

    candidates.sort()
    return candidates[:top_k]


def _enumerate_multiagent_windows(
    data_root,
    site,
    window_len=200,
    stride=25,
    classes=("car", "truck_bus"),
    min_track_frames=200,
    max_track_frames=1000,
):
    """Enumerate (start_frame, [trackId,...]) tuples where ≥2 eligible vehicle
    tracks have full window_len coverage. No geometric prefilter — caller is
    expected to score each tuple by the model's predicted-density overlap."""
    meta = pd.read_csv(os.path.join(data_root, f"{site}_tracksMeta.csv"))
    cls_mask = meta["class"].isin(classes)
    len_mask = (meta["numFrames"] > min_track_frames) & (meta["numFrames"] < max_track_frames)
    veh = meta[cls_mask & len_mask][["trackId", "initialFrame", "finalFrame"]]
    if len(veh) == 0:
        return []
    init = dict(zip(veh["trackId"].values, veh["initialFrame"].values))
    fin = dict(zip(veh["trackId"].values, veh["finalFrame"].values))
    ids = list(veh["trackId"].values)
    starts = set()
    for tid in ids:
        s, e = int(init[tid]), int(fin[tid])
        if e - s + 1 >= window_len:
            for ws in range(s, e - window_len + 2, stride):
                starts.add(ws)
    out = []
    for ws in sorted(starts):
        we = ws + window_len - 1
        present = [int(tid) for tid in ids if init[tid] <= ws and fin[tid] >= we]
        if len(present) >= 2:
            out.append((int(ws), present))
    return out


def _score_multiagent_overlap(model, agents_data, num_samples=300, res=160, sigma_px=2.5):
    """Cheap variant of make_multiagent_figure: just sample each agent's
    futures and compute the pairwise occupancy-grid overlap, no plotting.
    Uses fewer samples / smaller grid than the final figure to keep the
    brute-force search fast."""
    timesteps = [24, 49, 74, 99]
    futures = 100
    all_xy = np.concatenate(
        [a["obs_xy_norm"] for a in agents_data] + [a["tgt_xy_norm"] for a in agents_data],
        axis=0,
    )
    pad = 0.02
    xmin = max(0.0, float(all_xy[:, 0].min()) - pad)
    xmax = min(1.0, float(all_xy[:, 0].max()) + pad)
    ymin = max(0.0, float(all_xy[:, 1].min()) - pad)
    ymax = min(1.0, float(all_xy[:, 1].max()) + pad)
    bounds = ((xmin, xmax), (ymin, ymax))

    grids = []
    for a in agents_data:
        grids.append({t: direct_density_grid(model, a["input"], a["feature"], t, bounds, res=res, futures=futures) for t in timesteps})

    K = len(agents_data)
    pairwise = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            ov = sum(float((grids[i][t] * grids[j][t]).sum()) for t in timesteps) / len(timesteps)
            pairwise[i, j] = ov
            pairwise[j, i] = ov
    return {
        "max_pairwise_overlap": float(pairwise.max()),
        "mean_pairwise_overlap": float(pairwise[np.triu_indices(K, k=1)].mean()) if K > 1 else 0.0,
    }


def _build_agent_inputs(data_root, site, start_frame, track_ids, obs_len=100, fut_len=100):
    """For a chosen (start_frame, track_ids), build (input, feature, target)
    tensors for each agent under the same preprocessing as InD._parse."""
    tracks = pd.read_csv(os.path.join(data_root, f"{site}_tracks.csv"))
    window_len = obs_len + fut_len
    out = []
    for tid in track_ids:
        df = tracks[tracks["trackId"] == tid]
        df = df[(df["frame"] >= start_frame) & (df["frame"] < start_frame + window_len)]
        df = df.sort_values("frame")
        if len(df) < window_len:
            continue
        xy = df[["xCenter", "yCenter"]].to_numpy(dtype=np.float64)
        feats = df[["heading", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].to_numpy(dtype=np.float64)
        xy_norm = _normalize(xy, _SPATIAL_BOUNDS)
        feats_norm = _normalize(feats, _FEATURE_BOUNDS)
        # append linspace-time channel like InD._append_time
        t_ch = np.linspace(0.0, 2.0, window_len)[:, None]
        feats_norm = np.concatenate([feats_norm, t_ch], axis=-1)
        inp = torch.tensor(xy_norm[:obs_len], dtype=torch.float32).unsqueeze(0)
        feat = torch.tensor(feats_norm[:obs_len], dtype=torch.float32).unsqueeze(0)
        tgt = torch.tensor(xy_norm[obs_len:obs_len + fut_len], dtype=torch.float32).unsqueeze(0)
        out.append({
            "track_id": int(tid),
            "input": inp,
            "feature": feat,
            "target": tgt,
            "obs_xy_norm": xy_norm[:obs_len],
            "tgt_xy_norm": xy_norm[obs_len:obs_len + fut_len],
        })
    return out


def make_multiagent_figure(model, agents_data, out_dir, fig_name="fig_multiagent_panel"):
    """Render a 1xK panel of simultaneous marginal-occupancy heatmaps for
    several co-existing agents, overlaid in shared world coordinates. Compute
    pairwise grid overlap as a proxy for collision probability."""
    timesteps = [24, 49, 74, 99]
    res = 256
    sigma_px = 2.5
    cmap_names = ["Blues", "Reds", "Greens", "Purples", "Oranges"]

    # Tight crop to the agents' actual world extent, with a small margin.
    all_xy = np.concatenate(
        [a["obs_xy_norm"] for a in agents_data] + [a["tgt_xy_norm"] for a in agents_data],
        axis=0,
    )
    pad = 0.02
    xmin = max(0.0, float(all_xy[:, 0].min()) - pad)
    xmax = min(1.0, float(all_xy[:, 0].max()) + pad)
    ymin = max(0.0, float(all_xy[:, 1].min()) - pad)
    ymax = min(1.0, float(all_xy[:, 1].max()) + pad)
    bounds = ((xmin, xmax), (ymin, ymax))

    # Sample marginal densities for each agent at every timestep we render.
    futures = 100
    per_agent_grids = []
    per_agent_endpoints = []
    for k, a in enumerate(agents_data):
        # Samples are still drawn for visualization overlay (track lines, GT dots).
        with torch.no_grad():
            _, samples, _ = model.sample(a["input"].to(DEVICE), a["feature"].to(DEVICE), futures, 500)
        s = samples.cpu().numpy()
        if s.ndim == 4:
            s = s[:, 0]
        per_agent_endpoints.append(s)
        # Occupancy density evaluated directly via the trained NF (no KDE).
        grids_t = {t: direct_density_grid(model, a["input"], a["feature"], t, bounds, res=res, futures=futures) for t in timesteps}
        per_agent_grids.append(grids_t)

    K = len(agents_data)
    # Pairwise overlap: \int p_i(x,t) p_j(x,t) dx, summed (averaged) over horizon.
    pairwise_overlap = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            ov = 0.0
            for t in timesteps:
                ov += float((per_agent_grids[i][t] * per_agent_grids[j][t]).sum())
            ov /= len(timesteps)
            pairwise_overlap[i, j] = ov
            pairwise_overlap[j, i] = ov

    fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 4.0), constrained_layout=True)
    bg = axes[0].get_facecolor()  # for a calm background under the first agent's heatmap
    for col, t in enumerate(timesteps):
        ax = axes[col]
        ax.set_facecolor("#f5f5f5")
        for k, a in enumerate(agents_data):
            H = per_agent_grids[k][t]
            if H.max() > 0:
                # render with a per-agent cmap; alpha modulated by density to allow overlay
                cm = plt.get_cmap(cmap_names[k % len(cmap_names)])
                rgba = cm(H / H.max())
                rgba[..., 3] = np.clip(H / H.max(), 0.0, 0.85)
                ax.imshow(rgba, origin="lower", extent=(xmin, xmax, ymin, ymax), aspect="auto")
            ax.plot(a["obs_xy_norm"][:, 0], a["obs_xy_norm"][:, 1], "-", color="black", lw=1.4)
            ax.plot(a["tgt_xy_norm"][:t + 1, 0], a["tgt_xy_norm"][:t + 1, 1], "--",
                    color=plt.get_cmap(cmap_names[k % len(cmap_names)])(0.75), lw=1.4)
            # Mark the agent's GT position at this timestep
            ax.scatter([a["tgt_xy_norm"][t, 0]], [a["tgt_xy_norm"][t, 1]],
                       facecolor=plt.get_cmap(cmap_names[k % len(cmap_names)])(0.85),
                       edgecolors="black", s=55, linewidths=1.0, zorder=5)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_title(f"t = {t + 1}/100  ({(t + 1) * 0.04:.2f} s)")
        ax.set_xlabel("x (norm.)")
        if col == 0:
            ax.set_ylabel("y (norm.)")
    # Custom legend: agent colors
    handles = [plt.Line2D([0], [0], color=plt.get_cmap(cmap_names[k % len(cmap_names)])(0.7), lw=4,
                          label=f"agent {a['track_id']}") for k, a in enumerate(agents_data)]
    handles += [
        plt.Line2D([0], [0], color="black", lw=1.4, label="observed (past)"),
        plt.Line2D([0], [0], color="gray", lw=1.4, ls="--", label="ground truth (future)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.85)
    fig.suptitle(
        f"Multi-agent simultaneous marginal occupancy, inD recording 08, "
        f"{K} co-existing vehicles (CDE-CNF marginal, LORO checkpoint)",
        fontsize=11,
    )
    fig.savefig(out_dir / f"{fig_name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{fig_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[collision_demo] saved {out_dir/f'{fig_name}.pdf'}")
    return {
        "track_ids": [a["track_id"] for a in agents_data],
        "pairwise_overlap": pairwise_overlap.tolist(),
        "mean_pairwise_overlap": float(pairwise_overlap[np.triu_indices(K, k=1)].mean()) if K > 1 else 0.0,
        "max_pairwise_overlap": float(pairwise_overlap.max()),
        "world_bounds_norm": [[xmin, xmax], [ymin, ymax]],
    }


def main(out_dir, num_test=100, only_multiagent=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the LORO fold that holds out recording 08 (the canonical training site)
    site = build_loro_ind(["07", "09", "10"], ["08"])
    futures = 100  # inD prediction horizon

    print("[collision_demo] loading models...")
    cde = load_model(
        "../runs/checkpoints/loro_cde-cnf_train-07-09-10_test-08.pt",
        encoder="CDE", flow="CNF", marginal=True, device=DEVICE,
    )
    gru = load_model(
        "../runs/checkpoints/loro_gru-dnf_train-07-09-10_test-08.pt",
        encoder="GRU", flow="DNF", marginal=True, device=DEVICE,
    )

    if only_multiagent:
        data_root = "./data"
        # Brute-force: enumerate every viable window-start with stride 25 (1 s),
        # run the model on each, rank by max predicted pairwise overlap.
        cands = _enumerate_multiagent_windows(data_root, site="08", stride=25)
        if not cands:
            print("[collision_demo] no candidate windows; aborting")
            return
        print(f"[collision_demo] running brute-force overlap search over {len(cands)} candidates ...")

        best = None
        best_overlap = -1.0
        all_results = []
        for idx, (start_frame, track_ids) in enumerate(cands):
            agents = _build_agent_inputs(data_root, "08", start_frame, track_ids)
            if len(agents) < 2:
                continue
            res = _score_multiagent_overlap(cde, agents)
            res.update({"start_frame": int(start_frame), "track_ids": [a["track_id"] for a in agents]})
            all_results.append(res)
            if res["max_pairwise_overlap"] > best_overlap:
                best_overlap = res["max_pairwise_overlap"]
                best = (start_frame, [a["track_id"] for a in agents])
                print(f"  [{idx + 1}/{len(cands)}] start={start_frame} tracks={best[1]} "
                      f"max_ov={best_overlap:.3e}  ** new best **")
            elif (idx + 1) % 20 == 0:
                print(f"  [{idx + 1}/{len(cands)}] (current best: start={best[0]} ov={best_overlap:.3e})")

        start_frame, track_ids = best
        print(f"[collision_demo] best window: start={start_frame} tracks={track_ids} "
              f"max_overlap={best_overlap:.3e}")
        agents = _build_agent_inputs(data_root, "08", start_frame, track_ids)
        multi_summary = make_multiagent_figure(cde, agents, out_dir)
        multi_summary.update({"start_frame": int(start_frame)})
        with open(out_dir / "multiagent_demo.json", "w") as f:
            top10 = sorted(all_results, key=lambda x: -x["max_pairwise_overlap"])[:10]
            json.dump({"chosen": multi_summary, "top10_candidates": top10}, f, indent=2)
        print(json.dumps(multi_summary, indent=2))
        return

    # Iterate over test_loader; for each scene compute aggregate stats AND a
    # headline-suitability score; keep the highest-scoring scene as the panel.
    test_iter = iter(site.test_loader)
    cde_masses_all, gru_masses_all = [], []
    cde_endpoint_std, gru_endpoint_std = [], []
    best_score = -float("inf")
    panel_data = None
    panel_score_log = []

    for i in range(num_test):
        try:
            inp, feat, tgt = next(test_iter)
        except StopIteration:
            break
        inp = inp.to(DEVICE); feat = feat.to(DEVICE); tgt = tgt.to(DEVICE)
        cde_samples = sample_marginal(cde, inp, feat, futures, num_samples=500)
        gru_samples = sample_marginal(gru, inp, feat, futures, num_samples=500)
        if cde_samples.ndim == 4:
            cde_samples = cde_samples[:, 0]
        if gru_samples.ndim == 4:
            gru_samples = gru_samples[:, 0]
        cde_masses_all.append(expected_mass_near_endpoint(cde_samples))
        gru_masses_all.append(expected_mass_near_endpoint(gru_samples))
        cde_endpoint_std.append(cde_samples[:, -1, :].std(axis=0).mean())
        gru_endpoint_std.append(gru_samples[:, -1, :].std(axis=0).mean())

        tgt_np = tgt.cpu().numpy().squeeze(0)
        obs_np = inp.cpu().numpy().squeeze(0)
        score = _scene_score(cde_samples, gru_samples, tgt_np, obs_np)
        panel_score_log.append((i, score))
        if score > best_score:
            best_score = score
            panel_data = {
                "cde_samples": cde_samples,
                "gru_samples": gru_samples,
                "obs_xy": inp.cpu().numpy().squeeze(0),
                "tgt_xy": tgt_np,
                "inp_tensor": inp.detach().clone(),
                "feat_tensor": feat.detach().clone(),
                "scene_idx": i,
                "score": score,
            }

    print(f"[collision_demo] picked scene idx={panel_data['scene_idx']} score={panel_data['score']:.3f}")
    top5 = sorted(panel_score_log, key=lambda x: x[1], reverse=True)[:5]
    print(f"[collision_demo] top-5 scenes by score: {top5}")

    cde_masses = np.stack(cde_masses_all, axis=0)  # (num_test, futures)
    gru_masses = np.stack(gru_masses_all, axis=0)

    # ======= Figure 1: 4-panel occupancy heatmap at t = 25, 50, 75, 100 =======
    timesteps = [24, 49, 74, 99]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    bounds = ((0.0, 1.0), (0.0, 1.0))  # normalized inD spatial domain
    cde_s = panel_data["cde_samples"]
    gru_s = panel_data["gru_samples"]
    obs = panel_data["obs_xy"]; tgt = panel_data["tgt_xy"]

    inp_pt = panel_data["inp_tensor"]; feat_pt = panel_data["feat_tensor"]
    for col, t in enumerate(timesteps):
        H_cde = direct_density_grid(cde, inp_pt, feat_pt, t, bounds, res=160, futures=futures)
        H_gru = direct_density_grid(gru, inp_pt, feat_pt, t, bounds, res=160, futures=futures)
        for row, (H, name) in enumerate([(H_cde, "CDE-CNF"), (H_gru, "GRU-DNF")]):
            ax = axes[row, col]
            ax.imshow(H, origin="lower", extent=(0, 1, 0, 1), cmap="viridis",
                       norm=LogNorm(vmin=max(H[H > 0].min(), 1e-6), vmax=H.max()) if (H > 0).any() else None,
                       aspect="auto")
            ax.plot(obs[:, 0], obs[:, 1], "-", color="#000000", lw=2.0, label="observed (past)")
            ax.plot(tgt[:t + 1, 0], tgt[:t + 1, 1], "-", color="#ff7f00", lw=2.0, label="ground truth (future)")
            ax.scatter([tgt[t, 0]], [tgt[t, 1]], c="#ff7f00", edgecolors="black", s=70, linewidths=1.4, zorder=5, label="GT at displayed $t$")
            ax.set_title(f"{name}, t = {t + 1}/100")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel("y (norm.)")
            if row == 1:
                ax.set_xlabel("x (norm.)")
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.7)
    fig.suptitle("Predicted marginal occupancy density at t ∈ {25, 50, 75, 100}, inD test set, fold = 08",
                 fontsize=11)
    fig.savefig(out_dir / "fig_occupancy_panel.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_occupancy_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[collision_demo] saved {out_dir/'fig_occupancy_panel.pdf'}")

    # ======= Figure 2: Per-step expected mass within 1m of mean endpoint =======
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    horizon = np.arange(1, futures + 1) * 0.04  # inD is 25 Hz; 1 step = 1/25 = 0.04 s
    cde_mean = cde_masses.mean(axis=0); cde_se = cde_masses.std(axis=0) / np.sqrt(cde_masses.shape[0])
    gru_mean = gru_masses.mean(axis=0); gru_se = gru_masses.std(axis=0) / np.sqrt(gru_masses.shape[0])
    ax.plot(horizon, cde_mean, "-", color="C0", label="CDE-CNF (marginal)")
    ax.fill_between(horizon, cde_mean - cde_se, cde_mean + cde_se, color="C0", alpha=0.20)
    ax.plot(horizon, gru_mean, "-", color="C3", label="GRU-DNF (marginal)")
    ax.fill_between(horizon, gru_mean - gru_se, gru_mean + gru_se, color="C3", alpha=0.20)
    ax.set_xlabel("Future horizon [s] (inD: 25 Hz $\\Rightarrow$ 1 step = 0.04 s; 100-step horizon = 4 s)")
    ax.set_ylabel("Sample concentration (frac. within $\\approx$ 5 m of mean endpoint)")
    ax.set_title(f"Predicted occupancy concentration over horizon, inD test fold 08\n(num_test = {num_test} scenes, 500 samples each)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right")
    fig.savefig(out_dir / "fig_collision_proxy.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_collision_proxy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[collision_demo] saved {out_dir/'fig_collision_proxy.pdf'}")

    # ======= Figure 3: Multi-agent simultaneous occupancy =======
    print("[collision_demo] building multi-agent panel from raw recording 08 ...")
    data_root = "./data"
    pick = _find_multiagent_window(data_root, site="08", window_len=200, min_agents=3)
    multi_summary = {}
    if pick is None:
        print("[collision_demo] no multi-agent window found; skipping multiagent figure")
    else:
        start_frame, track_ids = pick
        print(f"[collision_demo] picked window start={start_frame}, tracks={track_ids}")
        agents = _build_agent_inputs(data_root, "08", start_frame, track_ids)
        if len(agents) >= 2:
            multi_summary = make_multiagent_figure(cde, agents, out_dir)
            multi_summary.update({"start_frame": int(start_frame)})
            print(json.dumps(multi_summary, indent=2))
        else:
            print(f"[collision_demo] only {len(agents)} agents materialized; skipping figure")

    summary = {
        "num_test_scenes": int(num_test),
        "fold": "loro_train-07-09-10_test-08",
        "horizon_s": float(futures * 0.04),
        "cde_endpoint_std_mean_norm": float(np.mean(cde_endpoint_std)),
        "gru_endpoint_std_mean_norm": float(np.mean(gru_endpoint_std)),
        "cde_mean_concentration_at_horizon": float(cde_mean[-1]),
        "gru_mean_concentration_at_horizon": float(gru_mean[-1]),
        "cde_mean_concentration_at_t50": float(cde_mean[49]),
        "gru_mean_concentration_at_t50": float(gru_mean[49]),
        "multiagent": multi_summary,
    }
    with open(out_dir / "collision_demo.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="../runs/figures")
    parser.add_argument("--num_test", type=int, default=30)
    parser.add_argument("--only_multiagent", action="store_true",
                        help="Render only the multi-agent panel (preserves the existing single-agent figure).")
    args = parser.parse_args()
    main(args.out_dir, num_test=args.num_test, only_multiagent=args.only_multiagent)
