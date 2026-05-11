"""Render the metric-sensitivity curve figure and the sampling-ablation table
   from the JSONs produced by inference_experiments.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = Path("../runs")
PAPER_DIR = Path("../original_submission")


def make_metric_sensitivity_figure():
    data = json.loads((RUNS / "metric_sensitivity.json").read_text())
    Ns = sorted(int(k) for k in data.keys())
    rmse = [data[str(n)]["rmse"] for n in Ns]
    crps = [data[str(n)]["crps"] for n in Ns]
    mad = [data[str(n)]["min_ade"] for n in Ns]
    mfd = [data[str(n)]["min_fde"] for n in Ns]

    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    ax.plot(Ns, mad, "o-", label="minADE", color="C0")
    ax.plot(Ns, mfd, "s-", label="minFDE", color="C1")
    ax.plot(Ns, rmse, "^-", label="RMSE", color="C2")
    ax.plot(Ns, crps, "d-", label="CRPS", color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("Number of samples per scene, $N$")
    ax.set_ylabel("Metric value (m for RMSE, dimensionless for CRPS)")
    ax.set_title("Metric sensitivity to sample count $N$ on inD test set\n(CDE-CNF marginal, single-session 75/25 split, seed 11)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right")
    out = PAPER_DIR / "fig_metric_sensitivity.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "fig_metric_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    print(json.dumps(data, indent=2))


def render_sampling_ablation_table_latex():
    data = json.loads((RUNS / "sampling_ablation.json").read_text())
    rows = []
    name_label = {
        "naive": "Naive ($N{=}20$)",
        "top5": "Top-$k$ ($k{=}5$)",
        "top10": "Top-$k$ ($k{=}10$)",
        "top20": "Top-$k$ ($k{=}20$)",
        "map": "MAP path",
        "viterbi": "Smoothness-reg.",
    }
    for key in ["naive", "top5", "top10", "top20", "map", "viterbi"]:
        if key not in data:
            continue
        d = data[key]
        rows.append(
            f"{name_label[key]} & {d['min_ade']:.3f} & {d['min_fde']:.3f} & "
            f"{d['endpoint_std']:.2f} & {d['jerk']:.4f} \\\\"
        )
    table = (
        "\\begin{table}[!t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Trajectory-assembly strategy comparison on inD (CDE-CNF marginal, single-session 75/25 split, $N=20$ output trajectories per scene, $50$ test scenes). Endpoint dispersion $\\sigma_{\\text{end}}$ is the per-scene standard deviation of the endpoint across the $20$ output trajectories, in metres. Mean per-step jerk is in normalized units (lower means smoother).}\n"
        "\\label{tab:assembly_strategies}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "\\textbf{Strategy} & \\textbf{minADE} & \\textbf{minFDE} & "
        "$\\sigma_{\\text{end}}$ \\textbf{(m)} & \\textbf{Jerk} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    out = PAPER_DIR / "tab_assembly_strategies.tex"
    out.write_text(table)
    print(f"saved {out}")
    print(table)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    make_metric_sensitivity_figure()
    render_sampling_ablation_table_latex()
