"""
Exploratory data analysis on the macrocycle dataset. Runs on pandas/matplotlib
only (no RDKit/torch needed) so it works anywhere, including in the sandbox
this project was authored in.

Run:
    python eda.py --csv data/All.csv --out_dir outputs/eda
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

PROPS = ["MW", "LogP", "HBA", "HBD", "PSA", "RotB"]
CLASS_ORDER = None  # filled in after load, sorted by frequency


def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    if "Unnamed: 5" in df.columns:
        df = df.drop(columns=["Unnamed: 5"])
    df["M_type"] = df["M_type"].astype(str).str.strip()
    df["M_type"] = df["M_type"].replace({"Cyclic peptide": "Cyclic Peptide"})
    df = df.dropna(subset=["SMILES"]).reset_index(drop=True)
    return df


def plot_class_balance(df, out_dir):
    counts = df["M_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("viridis", len(counts)))
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Class balance — M_type (log scale reveals the imbalance)")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, counts.values):
        pct = 100 * val / len(df)
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_balance.png"), dpi=150)
    plt.close(fig)


def plot_property_distributions(df, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, PROPS):
        vals = df[col].dropna()
        # clip extreme outliers just for display so the histogram is readable
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        sns.histplot(vals.clip(lo, hi), bins=50, ax=ax, color="teal", kde=True)
        ax.set_title(f"{col}  (median={vals.median():.1f})")
    fig.suptitle("Property distributions (1st-99th percentile clipped for display)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "property_distributions.png"), dpi=150)
    plt.close(fig)


def plot_property_by_class(df, out_dir):
    global CLASS_ORDER
    CLASS_ORDER = df["M_type"].value_counts().index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flat, PROPS):
        vals = df[col]
        lo, hi = vals.quantile(0.02), vals.quantile(0.98)
        d = df.copy()
        d[col] = d[col].clip(lo, hi)
        sns.boxplot(data=d, x="M_type", y=col, order=CLASS_ORDER, ax=ax,
                    palette="viridis", showfliers=False)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=40)
        ax.set_xlabel("")
    fig.suptitle("Property distribution by molecule class (2nd-98th percentile clipped)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "property_by_class.png"), dpi=150)
    plt.close(fig)


def plot_mw_logp_scatter(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    d = df[(df["MW"] < df["MW"].quantile(0.99)) & (df["LogP"].abs() < 20)]
    sns.scatterplot(data=d, x="MW", y="LogP", hue="M_type", alpha=0.35, s=12,
                     ax=ax, palette="tab10", linewidth=0)
    ax.set_title("Molecular weight vs LogP by class")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mw_vs_logp.png"), dpi=150)
    plt.close(fig)


def plot_smiles_length(df, out_dir):
    lengths = df["SMILES"].str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(lengths.clip(upper=lengths.quantile(0.99)), bins=60, ax=ax, color="darkorange")
    ax.axvline(200, color="red", linestyle="--", label="max_len=200 cutoff used in data_prep.py")
    dropped = (lengths > 200).sum()
    ax.set_title(f"SMILES string length distribution\n"
                 f"({dropped} molecules / {dropped/len(df):.1%} exceed max_len=200 and are dropped)")
    ax.set_xlabel("SMILES length (characters)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "smiles_length.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[PROPS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Property correlation matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "property_correlation.png"), dpi=150)
    plt.close(fig)


def plot_max_phase(df, out_dir):
    if "Max_Phase" not in df.columns:
        return
    phase = df["Max_Phase"].astype(str).str.strip().str.lower()
    phase = phase.replace({"phase1": "phase 1"})
    counts = phase.value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=sns.color_palette("magma", len(counts)))
    ax.set_yscale("log")
    ax.set_title("Clinical development stage (log scale)\n"
                  "— most molecules are preclinical; very few reach approval")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "max_phase.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/All.csv")
    ap.add_argument("--out_dir", default="outputs/eda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_clean(args.csv)
    print(f"Loaded {len(df)} molecules")

    plot_class_balance(df, args.out_dir)
    plot_property_distributions(df, args.out_dir)
    plot_property_by_class(df, args.out_dir)
    plot_mw_logp_scatter(df, args.out_dir)
    plot_smiles_length(df, args.out_dir)
    plot_correlation_heatmap(df, args.out_dir)
    plot_max_phase(df, args.out_dir)

    print(f"Saved 7 plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
