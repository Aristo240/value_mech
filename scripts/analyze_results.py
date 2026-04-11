#!/usr/bin/env python3
"""Cross-model analysis and figure generation.

Reads all outputs/*/results.json files and produces:
  - figures/b1_projection_ratio.png    — bar chart of projection ratios vs random baseline
  - figures/b2_steering_curves.png     — steering score vs alpha for each model × pilot value
  - figures/b3_angle_distribution.png  — polar plot of empirical value angles (Qwen)
  - figures/p5_cosine_divergence.png   — per-value cosine(user, asst) across models
  - figures/pvq_audit.png              — PVQ top-1 in-option rate across models
  - figures/summary_table.png          — pass/fail heatmap

Run:
  python scripts/analyze_results.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# --- Data loading -----------------------------------------------------------

MODEL_DIRS = [
    ("outputs", "Qwen-1.5B"),
    ("outputs/llama8", "Llama-8B"),
    ("outputs/gemma9", "Gemma-9B"),
    ("outputs/gemma27", "Gemma-27B"),
    ("outputs/llama70", "Llama-70B"),
]

PILOTS = [
    "pilot_b1_attractor",
    "pilot_b2_steering",
    "pilot_b3_gradient",
    "pilot_5_self_other",
    "pilot_5_control",
]


def load_results() -> dict[str, dict[str, dict]]:
    """Return {model_name: {pilot_name: results_dict}}."""
    data = {}
    for base, name in MODEL_DIRS:
        data[name] = {}
        for pilot in PILOTS:
            path = os.path.join(base, pilot, "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    data[name][pilot] = json.load(f)
    return data


# --- Figure 1: B1 Projection Ratios ----------------------------------------

def plot_b1_projection(data: dict):
    models, ratios, nulls, ci_los, ci_his = [], [], [], [], []
    for name, pilots in data.items():
        if "pilot_b1_attractor" not in pilots:
            continue
        r = pilots["pilot_b1_attractor"]
        models.append(name)
        ratios.append(r["projection_norm_ratio"])
        rb = r.get("random_baseline", {})
        nulls.append(rb.get("mean_ratio", 0))
        lo, hi = r.get("projection_norm_ratio_ci95", [0, 0])
        ci_los.append(lo)
        ci_his.append(hi)

    if not models:
        return

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, ratios, w, label="Observed ratio", color="#2196F3",
                   yerr=[[r - lo for r, lo in zip(ratios, ci_los)],
                         [hi - r for r, hi in zip(ratios, ci_his)]],
                   capsize=5, ecolor="gray")
    bars2 = ax.bar(x + w / 2, nulls, w, label="Random baseline (mean)", color="#BDBDBD")
    ax.axhline(0.30, ls="--", color="red", alpha=0.6, label="Pass threshold (0.30)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||proj|| / ||attractor||")
    ax.set_title("B1: Attractor Projection onto Value Subspace")
    ax.legend()
    ax.set_ylim(0, 1.1)

    for i, (r, p) in enumerate(zip(ratios, [data[m]["pilot_b1_attractor"].get("random_baseline", {}).get("p_value", 1) for m in models])):
        ax.text(i - w / 2, r + 0.03, f"p={p:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "b1_projection_ratio.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'b1_projection_ratio.png'}")


# --- Figure 2: B2 Steering Curves ------------------------------------------

def plot_b2_steering(data: dict):
    models_with_b2 = [(name, pilots["pilot_b2_steering"])
                       for name, pilots in data.items()
                       if "pilot_b2_steering" in pilots]
    if not models_with_b2:
        return

    n_models = len(models_with_b2)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    for col, (name, r) in enumerate(models_with_b2):
        ax = axes[0, col]
        alphas = r["alphas"]
        for pv in r["per_value"]:
            vi = pv["value_index"]
            vname = pv["value_name"]
            passed = pv.get("passes_decision_rule", False)
            ls = "-" if passed else "--"
            marker = "o" if passed else "x"
            ax.plot(alphas, pv["scores_base"], ls=ls, marker=marker, ms=4,
                    label=f"v{vi} base", alpha=0.6)
            ax.plot(alphas, pv["scores_instruct"], ls=ls, marker=marker, ms=4,
                    label=f"v{vi} instr", alpha=0.9)
        ax.set_xlabel("Steering alpha")
        ax.set_ylabel("Expected Likert score")
        ax.set_title(f"{name} — B2 Steering\n({r['n_passing']}/{len(r['per_value'])} pass)")
        ax.legend(fontsize=7, ncol=2, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "b2_steering_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'b2_steering_curves.png'}")


# --- Figure 3: B3 Angle Distribution (polar) -------------------------------

def plot_b3_angles(data: dict):
    for name, pilots in data.items():
        if "pilot_b3_gradient" not in pilots:
            continue
        r = pilots["pilot_b3_gradient"]
        angles = np.array(r["value_angles_deg"])
        inertia = np.array([pv["inertia"] for pv in r["per_value"]])
        names = [pv["value_name"] for pv in r["per_value"]]
        attractor_angle = r["attractor_angle_deg"]
        ad = r.get("angle_diagnostics", {})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                        subplot_kw={"projection": "polar"})
        # Left: value positions on the circle
        theta = np.radians(angles)
        ax1.scatter(theta, np.ones_like(theta), c=inertia, cmap="RdBu_r",
                    s=80, edgecolors="k", linewidths=0.5, zorder=5)
        for i, nm in enumerate(names):
            short = nm.split(":")[0][:12]
            ax1.annotate(short, (theta[i], 1.12), fontsize=6, ha="center")
        ax1.plot([np.radians(attractor_angle)] * 2, [0, 1], "r-", lw=2,
                 label="Attractor", zorder=4)
        ax1.set_title(f"{name} — Empirical Value Angles\n"
                      f"(gap CV={ad.get('gap_cv', '?'):.1f}, "
                      f"{ad.get('n_clusters', '?')} clusters)")
        ax1.set_ylim(0, 1.3)
        ax1.set_yticks([])

        # Right: inertia vs angle with cosine fit
        cf = r["cosine_fit"]
        theta_fit = np.linspace(0, 2 * np.pi, 200)
        A, phi_deg, C = cf["amplitude"], cf["phase_deg"], 0
        # Recover C from data mean
        C = np.mean(inertia) - A * np.mean(np.cos(theta - np.radians(phi_deg)))
        fit_vals = A * np.cos(theta_fit - np.radians(phi_deg)) + C

        ax2.scatter(theta, np.abs(inertia) * 100 + 0.5, c=inertia, cmap="RdBu_r",
                    s=60, edgecolors="k", linewidths=0.5, zorder=5)
        ax2.set_title(f"Cosine Fit: R²={cf['r2']:.3f}, p={cf['permutation_p_value']:.3f}\n"
                      f"DECISION: {'PASS' if r['decision_rule_passed'] else 'FAIL'}")
        ax2.set_yticks([])

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "b3_angle_distribution.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {FIGURES_DIR / 'b3_angle_distribution.png'}")


# --- Figure 4: Pilot 5 Cosine Divergence -----------------------------------

def plot_p5_cosine(data: dict):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import SCHWARTZ_19

    models_with_p5 = [(name, pilots["pilot_5_self_other"])
                       for name, pilots in data.items()
                       if "pilot_5_self_other" in pilots]
    if not models_with_p5:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(SCHWARTZ_19))
    width = 0.8 / len(models_with_p5)
    colors = sns.color_palette("husl", len(models_with_p5))

    for i, (name, r) in enumerate(models_with_p5):
        cos_vals = r["test_5b_cos_per_value"]
        offset = (i - len(models_with_p5) / 2 + 0.5) * width
        ax.bar(x + offset, cos_vals, width, label=f"{name} (med={r['test_5b_median_cos']:.2f})",
               color=colors[i], alpha=0.8)

    ax.axhline(0.7, ls="--", color="red", alpha=0.5, label="Pass threshold (< 0.7)")
    ax.set_xticks(x)
    ax.set_xticklabels([v.split(":")[0][:10] for v in SCHWARTZ_19], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("cos(v_user, v_assistant)")
    ax.set_title("Pilot 5b: Per-Value Cosine Divergence (User vs Assistant)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "p5_cosine_divergence.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'p5_cosine_divergence.png'}")


# --- Figure 5: PVQ Audit ---------------------------------------------------

def plot_pvq_audit(data: dict):
    models, top1_rates, masses = [], [], []
    for name, pilots in data.items():
        for pilot_name in ["pilot_b2_steering", "pilot_b3_gradient"]:
            if pilot_name in pilots and "pvq_audit" in pilots[pilot_name]:
                audit = pilots[pilot_name]["pvq_audit"]
                if audit["total_scored"] > 0:
                    models.append(f"{name}\n({pilot_name.split('_')[-1]})")
                    top1_rates.append(audit["top1_in_options_frac"])
                    masses.append(audit["mean_option_prob_mass"])

    if not models:
        print("  Skipping PVQ audit plot (no data from fresh runs)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(models)), [r * 100 for r in top1_rates], color="#FF7043")
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("% of predictions")
    ax1.set_title("Top-1 Prediction Within {1..6}")
    ax1.set_ylim(0, 105)
    ax1.axhline(80, ls="--", color="green", alpha=0.5, label="80% (reliable)")
    ax1.legend()

    ax2.bar(range(len(models)), [m * 100 for m in masses], color="#42A5F5")
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("% probability mass")
    ax2.set_title("Mean Probability Mass on Option Tokens")
    ax2.set_ylim(0, 105)
    ax2.axhline(50, ls="--", color="green", alpha=0.5, label="50% (acceptable)")
    ax2.legend()

    fig.suptitle("PVQ Scoring Validity Audit", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pvq_audit.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'pvq_audit.png'}")


# --- Figure 6: Summary Heatmap ---------------------------------------------

def plot_summary_heatmap(data: dict):
    pilot_labels = ["B1\nAttractor", "B2\nSteering", "B3\nCosine", "Pilot 5\nSelf/Other", "P5 Control\n(at chance?)"]
    pilot_keys = ["pilot_b1_attractor", "pilot_b2_steering", "pilot_b3_gradient",
                   "pilot_5_self_other", "pilot_5_control"]
    model_names = [name for _, name in MODEL_DIRS if name in data and len(data[name]) > 0]
    n_rows = len(model_names)
    n_cols = len(pilot_keys)
    if n_rows == 0:
        print("  Skipping summary table (no results)")
        return

    # Build numeric matrix: 0=fail, 1=pass, 0.5=neutral, nan=missing
    matrix = np.full((n_rows, n_cols), np.nan)
    annotations = [["—" for _ in range(n_cols)] for _ in range(n_rows)]

    for i, name in enumerate(model_names):
        for j, pk in enumerate(pilot_keys):
            if pk not in data[name]:
                continue
            r = data[name][pk]
            passed = r.get("decision_rule_passed")
            if pk == "pilot_5_control":
                at_chance = r.get("shuffled_at_chance", None)
                if at_chance is True:
                    matrix[i][j] = 0.5
                    annotations[i][j] = "at chance"
                elif at_chance is False:
                    matrix[i][j] = 1.0
                    annotations[i][j] = "above\nchance"
            elif passed is True:
                matrix[i][j] = 1.0
                annotations[i][j] = "PASS"
            elif passed is False:
                matrix[i][j] = 0.0
                annotations[i][j] = "FAIL"

    # Build the table using matplotlib table — reliable across backends.
    color_map = {-1: "#E0E0E0", 0: "#EF5350", 0.5: "#FFE082", 1: "#66BB6A"}
    text_color_map = {-1: "#999999", 0: "white", 0.5: "black", 1: "white"}

    # Replace NaN with -1 for display
    display_matrix = np.where(np.isnan(matrix), -1.0, matrix)

    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.8 * n_rows))
    ax.axis("off")

    cell_text = annotations
    col_labels = ["B1 Attractor", "B2 Steering", "B3 Cosine", "Pilot 5", "P5 Control"]
    row_labels = model_names

    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)

    # Color cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = display_matrix[i][j]
            # table cell indices: row 0 is header, data rows start at 1
            cell = table[i + 1, j]
            cell.set_facecolor(color_map.get(val, "#E0E0E0"))
            cell.set_text_props(color=text_color_map.get(val, "black"),
                                fontweight="bold")
            cell.set_edgecolor("white")
            cell.set_linewidth(2)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#424242")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")
        cell.set_linewidth(2)

    # Style row labels
    for i in range(n_rows):
        cell = table[i + 1, -1]
        cell.set_facecolor("#F5F5F5")
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("white")
        cell.set_linewidth(2)

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight="bold", rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_title("Pilot Results Summary", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'summary_table.png'}")


# --- Statistical summary text ----------------------------------------------

def print_statistics(data: dict):
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    # B1 scaling trend
    b1_models = []
    for name, pilots in data.items():
        if "pilot_b1_attractor" in pilots:
            r = pilots["pilot_b1_attractor"]
            params = {"Qwen-1.5B": 1.5, "Llama-8B": 8, "Gemma-9B": 9, "Gemma-27B": 27}.get(name)
            if params:
                b1_models.append((params, r["projection_norm_ratio"], name))
    if b1_models:
        b1_models.sort()
        print("\nB1 — Projection ratio scales with model size:")
        for params, ratio, name in b1_models:
            print(f"  {name:12s} ({params:5.1f}B): ratio = {ratio:.3f}")
        if len(b1_models) >= 2:
            from scipy.stats import spearmanr
            sizes, ratios = zip(*[(p, r) for p, r, _ in b1_models])
            rho, p = spearmanr(sizes, ratios)
            print(f"  Spearman rho = {rho:.3f} (p = {p:.3f}) — "
                  f"{'significant' if p < 0.05 else 'not significant (N too small)'}")

    # B2 PVQ validity
    print("\nB2 — PVQ scoring validity issue:")
    for name, pilots in data.items():
        if "pilot_b2_steering" in pilots:
            r = pilots["pilot_b2_steering"]
            audit = r.get("pvq_audit", {})
            top1 = audit.get("top1_in_options_frac", "?")
            n_pass = r["n_passing"]
            total = len(r["per_value"])
            print(f"  {name:12s}: {n_pass}/{total} pass, PVQ top-1 in options = "
                  f"{top1:.0%}" if isinstance(top1, float) else f"  {name}: {n_pass}/{total} pass, PVQ audit = cached")

    # Pilot 5 cosine trend
    print("\nPilot 5b — Median cosine(user, assistant) by model:")
    for name, pilots in data.items():
        if "pilot_5_self_other" in pilots:
            r = pilots["pilot_5_self_other"]
            print(f"  {name:12s}: median_cos = {r['test_5b_median_cos']:.3f}")

    print("\n" + "=" * 70)


# --- Main -------------------------------------------------------------------

def main():
    # Ensure we run from the repo root regardless of where the script is called from.
    os.chdir(Path(__file__).resolve().parent.parent)
    print("Loading results...")
    data = load_results()
    available = {name: list(pilots.keys()) for name, pilots in data.items() if pilots}
    print(f"Found results for {len(available)} models:")
    for name, pilots in available.items():
        print(f"  {name}: {', '.join(p.replace('pilot_', '') for p in pilots)}")

    print("\nGenerating figures...")
    plot_b1_projection(data)
    plot_b2_steering(data)
    plot_b3_angles(data)
    plot_p5_cosine(data)
    plot_pvq_audit(data)
    plot_summary_heatmap(data)

    print_statistics(data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
