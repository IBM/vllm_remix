#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

"""
Visualize a cluster-tier mapping YAML file.

Usage:
    python plot_mapping.py [path/to/mapping.yaml]

If no argument is given, plots all files in the mapping/ folder and saves
them to mapping_plots/.
"""

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULT_DIR = Path(__file__).parent


def load_mapping(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data


def plot_mapping(yaml_path, ax=None, title=None, save_path=None):
    data = load_mapping(yaml_path)

    # Collect all (cluster, tier, job_id, layer) tuples
    records = []
    job_ids = []
    for entry in data.values():
        jid = entry["job id"]
        layer = entry["layer"]
        job_ids.append(jid)
        for cluster, tiers in entry["Cluster - Tier Mapping"].items():
            for tier in tiers:
                records.append((int(cluster), int(tier), jid, layer))

    all_clusters = sorted({r[0] for r in records})
    all_tiers = sorted({r[1] for r in records})
    unique_jobs = sorted(set(job_ids))

    # Map job_id → integer index for coloring
    job_to_idx = {jid: i for i, jid in enumerate(unique_jobs)}
    n_jobs = len(unique_jobs)

    # Build grid: rows=tiers (y), cols=clusters (x), value=job_idx (+1, 0=empty)
    c_min, c_max = min(all_clusters), max(all_clusters)
    t_min, t_max = min(all_tiers), max(all_tiers)
    grid = np.full((t_max - t_min + 1, c_max - c_min + 1), -1, dtype=int)

    for cluster, tier, jid, _ in records:
        grid[tier - t_min, cluster - c_min] = job_to_idx[jid]

    # Per-job metadata for legend (job_id → layer)
    job_layer = {}
    for entry in data.values():
        job_layer[entry["job id"]] = entry["layer"]

    # Colormap: one distinct color per job
    cmap = plt.get_cmap("tab20", n_jobs)
    color_matrix = np.zeros((*grid.shape, 4))
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            idx = grid[r, c]
            if idx >= 0:
                color_matrix[r, c] = cmap(idx)
            else:
                color_matrix[r, c] = [0.92, 0.92, 0.92, 1.0]  # empty = light grey

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(max(10, (c_max - c_min + 1) * 0.28), max(5, (t_max - t_min + 1) * 0.55))
        )

    ax.imshow(
        color_matrix,
        origin="lower",
        aspect="auto",
        extent=[c_min - 0.5, c_max + 0.5, t_min - 0.5, t_max + 0.5],
        interpolation="nearest",
    )

    # Grid lines
    for c in range(c_min, c_max + 2):
        ax.axvline(c - 0.5, color="white", lw=0.4, alpha=0.6)
    for t in range(t_min, t_max + 2):
        ax.axhline(t - 0.5, color="white", lw=0.4, alpha=0.6)

    # Annotate each occupied cell with job_id
    for cluster, tier, jid, _ in records:
        ax.text(
            cluster,
            tier,
            str(jid),
            ha="center",
            va="center",
            fontsize=5.5,
            color="white",
            fontweight="bold",
        )

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Tier")
    ax.set_xticks(all_clusters)
    ax.set_xticklabels(all_clusters, fontsize=6, rotation=90)
    ax.set_yticks(all_tiers)
    ax.set_yticklabels(all_tiers, fontsize=7)

    if title is None:
        title = Path(yaml_path).stem
    ax.set_title(title, fontsize=8)

    # Legend: job_id → layer
    patches = [
        mpatches.Patch(color=cmap(job_to_idx[jid]), label=f"job {jid}  (layer {job_layer[jid]})")
        for jid in unique_jobs
    ]
    ax.legend(
        handles=patches,
        fontsize=6,
        loc="upper right",
        ncol=max(1, n_jobs // 12),
        framealpha=0.85,
        title="job id (model layer)",
        title_fontsize=6,
    )

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")
        else:
            plt.show()
        plt.close()


def batch_all():
    mapping_dir = RESULT_DIR / "mapping"
    out_dir = RESULT_DIR / "mapping_plots"
    out_dir.mkdir(exist_ok=True)

    files = sorted(mapping_dir.glob("*.yaml"))
    print(f"Generating {len(files)} plots → {out_dir}")
    for f in files:
        save_path = out_dir / (f.stem + ".png")
        plot_mapping(f, save_path=save_path)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_mapping(sys.argv[1])
    else:
        batch_all()
