#!/usr/bin/env python3
"""KineScaper SED Robustness Analysis
====================================
Comparative analysis for 3 SED models on KineScaper-EV evaluation runs.

What it does:
- Loads per-sample SED results from results.csv (3 models)
- Merges KineScaper metadata (siren_class + SNR) via segment_id
- Handles duplicate metadata segment_id robustly (SNR averaged per segment_id)
- Aggregates metrics by (model, siren_class, SNR bin)
- Uses NaN-safe aggregation (skipna)
- Generates line plots with shaded ±std area for:
    - seg_precision
    - seg_f1
    - seg_balanced_accuracy

Binning:
- SNR bin width fixed to 6 dB

Output folder:
- results/sed/evaluation/analysis_kinescaper_robustness_6dB/

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = PROJECT_ROOT / "results" / "sed" / "evaluation"
OUTPUT_ROOT = EVAL_ROOT / "analysis_kinescaper_robustness_6dB"
PLOTS_DIR = OUTPUT_ROOT / "plots"
TABLES_DIR = OUTPUT_ROOT / "tables"

METADATA_JSON = Path("/mnt/ssd/Kinescaper_EV/dataset/json/metadata.json")
BIN_WIDTH_DB = 6.0

MODEL_RUNS = {
    "EPANNs finetuned AS-EV v2": EVAL_ROOT / "sed_KineScaper_EV_epanns_finetuned_AS-EV_v2" / "results.csv",
    "EPANNs unified EV": EVAL_ROOT / "sed_KineScaper_EV_epanns_unified" / "results.csv",
    "CLAP unified EV": EVAL_ROOT / "sed_KineScaper_EV_clap_unified" / "results.csv",
}

MODEL_COLORS = {
    "EPANNs finetuned AS-EV v2": "#1f77b4",  # blue
    "EPANNs unified EV": "#ff7f0e",          # orange
    "CLAP unified EV": "#2ca02c",            # green
}

METRICS = ["seg_precision", "seg_f1", "seg_balanced_accuracy"]
CLASS_ORDER = ["hi-lo", "phaser", "piercer", "rumbler", "two-tone", "wail", "yelp"]


# =============================================================================
# IO + MERGE
# =============================================================================
def load_metadata() -> pd.DataFrame:
    """Load KineScaper metadata and deduplicate segment_id robustly.

    Some segment_id values appear duplicated in metadata with different SNR values.
    We collapse duplicates by averaging SNR fields and keeping class (which is stable).
    """
    if not METADATA_JSON.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_JSON}")

    with open(METADATA_JSON, "r") as f:
        raw = json.load(f)["dataset_metadata"]

    meta = pd.DataFrame(raw)
    meta["segment_id"] = meta["filename"].str.replace(".wav", "", regex=False)

    # Deduplicate by segment_id: class is consistent, SNR may differ
    meta_dedup = (
        meta.groupby("segment_id", as_index=False)
        .agg(
            siren_class=("siren_class", "first"),
            snr_avg=("snr_avg", "mean"),
            snr_min=("snr_min", "mean"),
            snr_max=("snr_max", "mean"),
        )
    )

    return meta_dedup


def load_model_results() -> pd.DataFrame:
    """Load and stack KineScaper results for all configured models."""
    frames = []
    for model_name, csv_path in MODEL_RUNS.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing results.csv for model '{model_name}': {csv_path}")

        df = pd.read_csv(csv_path)
        df["model"] = model_name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    return data


def enrich_with_metadata(results_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Merge results with KineScaper class + SNR metadata."""
    merged = results_df.merge(
        metadata_df[["segment_id", "siren_class", "snr_avg", "snr_min", "snr_max"]],
        on="segment_id",
        how="left",
        validate="many_to_one",
    )

    missing = merged["siren_class"].isna().sum()
    if missing > 0:
        raise ValueError(f"Metadata merge incomplete: {missing} rows without siren_class")

    return merged


# =============================================================================
# AGGREGATION
# =============================================================================
def build_snr_bins(df: pd.DataFrame, width_db: float = BIN_WIDTH_DB) -> np.ndarray:
    """Create global SNR edges (aligned to bin width) over all merged rows."""
    min_snr = df["snr_avg"].min()
    max_snr = df["snr_avg"].max()

    start = np.floor(min_snr / width_db) * width_db
    end = np.ceil(max_snr / width_db) * width_db + width_db
    edges = np.arange(start, end + 1e-9, width_db)

    if len(edges) < 2:
        edges = np.array([start, start + width_db])

    return edges


def aggregate_metrics(df: pd.DataFrame, bin_edges: np.ndarray) -> pd.DataFrame:
    """Aggregate mean/std/count for metrics by model, class, and SNR bin.

    NaN handling: skipna (default behavior of pandas mean/std).
    """
    out = df.copy()
    out["snr_bin"] = pd.cut(out["snr_avg"], bins=bin_edges, right=False, include_lowest=True)

    grouped = out.groupby(["model", "siren_class", "snr_bin"], observed=False)

    rows = []
    for (model, siren_class, snr_bin), g in grouped:
        rec = {
            "model": model,
            "siren_class": siren_class,
            "snr_bin": str(snr_bin),
            "snr_bin_left": float(snr_bin.left),
            "snr_bin_right": float(snr_bin.right),
            "snr_bin_center": float((snr_bin.left + snr_bin.right) / 2.0),
            "group_count": int(len(g)),
        }

        for metric in METRICS:
            rec[f"{metric}_mean"] = g[metric].mean(skipna=True)
            rec[f"{metric}_std"] = g[metric].std(skipna=True)
            rec[f"{metric}_valid_n"] = int(g[metric].notna().sum())
            rec[f"{metric}_nan_n"] = int(g[metric].isna().sum())

        rows.append(rec)

    agg = pd.DataFrame(rows)

    # Keep class ordering stable
    agg["siren_class"] = pd.Categorical(agg["siren_class"], categories=CLASS_ORDER, ordered=True)
    agg = agg.sort_values(["siren_class", "model", "snr_bin_left"]).reset_index(drop=True)

    return agg


def aggregate_global_weighted(agg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate globally across classes, weighted by valid sample count per bin.

    For each (model, SNR bin), compute weighted mean/std for each metric using
    class-level valid_n as weights. This avoids over-weighting sparse class bins.
    """
    rows = []

    for (model, snr_bin, snr_left, snr_right, snr_center), g in agg.groupby(
        ["model", "snr_bin", "snr_bin_left", "snr_bin_right", "snr_bin_center"],
        observed=False,
    ):
        rec = {
            "model": model,
            "snr_bin": snr_bin,
            "snr_bin_left": snr_left,
            "snr_bin_right": snr_right,
            "snr_bin_center": snr_center,
        }

        for metric in METRICS:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            n_col = f"{metric}_valid_n"

            valid = g[(g[n_col] > 0) & (~g[mean_col].isna())].copy()
            if valid.empty:
                rec[f"{metric}_mean"] = np.nan
                rec[f"{metric}_std"] = np.nan
                rec[f"{metric}_valid_n"] = 0
                rec[f"{metric}_nan_n"] = int(g[f"{metric}_nan_n"].sum())
                continue

            w = valid[n_col].to_numpy(dtype=float)
            x = valid[mean_col].to_numpy(dtype=float)

            weighted_mean = np.average(x, weights=w)

            # Combine within-class std and between-class dispersion
            within_var = np.sum(w * (valid[std_col].fillna(0.0).to_numpy(dtype=float) ** 2)) / np.sum(w)
            between_var = np.sum(w * ((x - weighted_mean) ** 2)) / np.sum(w)
            weighted_std = float(np.sqrt(max(within_var + between_var, 0.0)))

            rec[f"{metric}_mean"] = float(weighted_mean)
            rec[f"{metric}_std"] = weighted_std
            rec[f"{metric}_valid_n"] = int(valid[n_col].sum())
            rec[f"{metric}_nan_n"] = int(g[f"{metric}_nan_n"].sum())

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values(["model", "snr_bin_left"]).reset_index(drop=True)
    return out


# =============================================================================
# PLOTTING
# =============================================================================
def plot_metric_by_class(agg: pd.DataFrame, metric: str, save_path: Path):
    """7-panel line plots (one per class), x=SNR bin center, y=metric mean ± std."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    model_order = list(MODEL_RUNS.keys())

    for i, siren_class in enumerate(CLASS_ORDER):
        ax = axes[i]
        class_df = agg[agg["siren_class"] == siren_class]

        for model in model_order:
            d = class_df[class_df["model"] == model].copy()
            d = d[d[f"{metric}_valid_n"] > 0].sort_values("snr_bin_center")
            if d.empty:
                continue

            x = d["snr_bin_center"].to_numpy()
            y = d[f"{metric}_mean"].to_numpy(dtype=float)
            y_std = d[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)

            color = MODEL_COLORS.get(model, None)
            ax.plot(x, y, marker="o", linewidth=2, label=model, color=color)
            ax.fill_between(
                x,
                np.clip(y - y_std, 0, 1),
                np.clip(y + y_std, 0, 1),
                alpha=0.15,
                color=color,
            )

        ax.set_title(f"{siren_class}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, frameon=False)

    # Hide remaining empty panels (9 grid for 7 classes)
    for j in range(len(CLASS_ORDER), len(axes)):
        axes[j].axis("off")

    metric_title = metric.replace("seg_", "").replace("_", " ").title()
    fig.suptitle(f"KineScaper Robustness – {metric_title} vs SNR (6 dB bins, mean ± std)", fontsize=15)
    fig.supxlabel("SNR bin center (dB)")
    fig.supylabel(metric_title)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(save_path, dpi=600, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_global_comparison(global_agg: pd.DataFrame, save_path: Path):
    """Create global 3-panel comparison plot (precision, f1, balanced accuracy)."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharex=True, sharey=True)
    model_order = list(MODEL_RUNS.keys())

    for ax, metric in zip(axes, METRICS):
        for model in model_order:
            d = global_agg[global_agg["model"] == model].copy()
            d = d[d[f"{metric}_valid_n"] > 0].sort_values("snr_bin_center")
            if d.empty:
                continue

            x = d["snr_bin_center"].to_numpy(dtype=float)
            y = d[f"{metric}_mean"].to_numpy(dtype=float)
            y_std = d[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)

            color = MODEL_COLORS.get(model, None)
            ax.plot(x, y, marker="o", linewidth=2.2, label=model, color=color)
            ax.fill_between(
                x,
                np.clip(y - y_std, 0, 1),
                np.clip(y + y_std, 0, 1),
                alpha=0.15,
                color=color,
            )

        ax.set_title(metric.replace("seg_", "").replace("_", " ").title())
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle("Global model comparison vs SNR (6 dB bins, weighted mean ± std)", fontsize=14)
    fig.supxlabel("SNR bin center (dB)")
    fig.supylabel("Metric value")

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(save_path, dpi=600, format="svg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("KineScaper SED Robustness Analysis (3 models)")
    print("=" * 80)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results and metadata...")
    results = load_model_results()
    meta = load_metadata()
    merged = enrich_with_metadata(results, meta)

    print(f"  Results rows: {len(results):,}")
    print(f"  Metadata rows (dedup): {len(meta):,}")
    print(f"  Merged rows: {len(merged):,}")

    edges = build_snr_bins(merged, BIN_WIDTH_DB)
    print(f"  SNR bins: {len(edges)-1} bins, width={BIN_WIDTH_DB} dB, range=[{edges[0]:.1f}, {edges[-1]:.1f})")

    agg = aggregate_metrics(merged, edges)
    global_agg = aggregate_global_weighted(agg)

    # Save tables
    merged_path = TABLES_DIR / "kinescaper_merged_per_sample.csv"
    agg_path = TABLES_DIR / "kinescaper_aggregated_by_class_snr_6dB.csv"
    global_path = TABLES_DIR / "kinescaper_global_weighted_by_snr_6dB.csv"
    merged.to_csv(merged_path, index=False)
    agg.to_csv(agg_path, index=False)
    global_agg.to_csv(global_path, index=False)

    print(f"Saved: {merged_path}")
    print(f"Saved: {agg_path}")
    print(f"Saved: {global_path}")

    # Plots
    for metric in METRICS:
        plot_path = PLOTS_DIR / f"lineplot_{metric}_vs_snr_6dB.svg"
        plot_metric_by_class(agg, metric, plot_path)
        print(f"Saved: {plot_path}")

    global_plot_path = PLOTS_DIR / "global_comparison_metrics_vs_snr_6dB.svg"
    plot_global_comparison(global_agg, global_plot_path)
    print(f"Saved: {global_plot_path}")

    # Compact JSON report
    report = {
        "bin_width_db": BIN_WIDTH_DB,
        "num_models": len(MODEL_RUNS),
        "num_rows_results": int(len(results)),
        "num_rows_merged": int(len(merged)),
        "snr_range": [float(edges[0]), float(edges[-1])],
        "metrics": METRICS,
        "model_runs": {k: str(v) for k, v in MODEL_RUNS.items()},
    }

    report_path = OUTPUT_ROOT / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
