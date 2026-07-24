"""
Plotting functions for generative-model evaluation. Deliberately has NO
torch/rdkit import so it can be unit-tested and reused independently of the
model (e.g. for quick exploratory plots of any generated-molecule CSV).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PLOT_PROPERTIES = ["MW", "LogP", "PSA", "RotB", "HBA", "HBD"]


def plot_property_distributions(real_df, gen_df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, prop in zip(axes.flat, PLOT_PROPERTIES):
        real_vals = real_df[prop].dropna() if prop in real_df.columns else pd.Series([], dtype=float)
        gen_vals = gen_df[prop].dropna() if prop in gen_df.columns else pd.Series([], dtype=float)
        combined = pd.concat([real_vals, gen_vals]) if len(gen_vals) else real_vals
        if len(combined) == 0:
            ax.set_title(f"{prop} (no data)")
            continue
        bins = np.histogram_bin_edges(combined, bins=40)
        if len(real_vals):
            ax.hist(real_vals, bins=bins, density=True, alpha=0.5, label="real (training set)", color="#4C72B0")
        if len(gen_vals):
            ax.hist(gen_vals, bins=bins, density=True, alpha=0.5, label="generated (valid only)", color="#DD8452")
        ax.set_title(prop)
        ax.set_ylabel("density")
    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Property distributions: real training set vs. generated molecules")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_validity_by_class(per_class_metrics, out_path):
    classes = list(per_class_metrics.keys())
    validity = [per_class_metrics[c]["validity"] for c in classes]
    novelty = [per_class_metrics[c]["novelty"] for c in classes]
    uniqueness = [per_class_metrics[c]["uniqueness"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, validity, width, label="validity", color="#55A868")
    ax.bar(x, uniqueness, width, label="uniqueness", color="#4C72B0")
    ax.bar(x + width, novelty, width, label="novelty", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Generation quality by molecule class\n(watch for rare classes lagging behind common ones)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_qed_sa(gen_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    titles = {"QED": "QED (drug-likeness, higher=better)",
              "SA": "SA score (synthetic accessibility, lower=easier)"}
    for ax, prop in zip(axes, ["QED", "SA"]):
        gen_vals = gen_df[prop].dropna() if prop in gen_df.columns else pd.Series([], dtype=float)
        if len(gen_vals) == 0:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(titles[prop])
            continue
        ax.hist(gen_vals, bins=30, alpha=0.7, color="#8172B2")
        ax.axvline(gen_vals.mean(), color="black", linestyle="--", linewidth=1,
                   label=f"mean={gen_vals.mean():.2f}")
        ax.set_title(titles[prop])
        ax.legend(fontsize=8)
    fig.suptitle("Generated-molecule quality scores")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_conditioning_fidelity(target_df, achieved_df, fidelity, properties, out_path):
    """
    Asked-for vs. actually-got, one panel per conditioning property, with the
    y=x identity line. Points hugging the diagonal = the conditioning vector is
    steering generation; a horizontal blob = the model is ignoring it and just
    emitting the class mean (which a raw MAE number alone would not reveal).
    """
    n = len(properties)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, prop in zip(axes, properties):
        if prop not in target_df.columns or prop not in achieved_df.columns:
            ax.set_visible(False)
            continue
        t = pd.to_numeric(target_df[prop], errors="coerce")
        a = pd.to_numeric(achieved_df[prop], errors="coerce")
        ok = t.notna() & a.notna()
        t, a = t[ok].values, a[ok].values
        if len(t) == 0:
            ax.set_visible(False)
            continue

        ax.scatter(t, a, s=8, alpha=0.25, color="#4C72B0", linewidth=0)
        lo = float(min(t.min(), a.min()))
        hi = float(max(t.max(), a.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect (y=x)")

        m = fidelity.get("per_property", {}).get(prop)
        if m:
            ax.set_title(
                f"{prop}\nMAE={m['mae']:.2f}  r={m['pearson_r']:.2f}  "
                f"gain={m['fidelity_gain']:.2f}x" if m["fidelity_gain"] else prop,
                fontsize=10,
            )
        else:
            ax.set_title(prop)
        ax.set_xlabel(f"requested {prop}")
        ax.set_ylabel(f"generated {prop}")
        ax.legend(fontsize=7, loc="upper left")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Conditioning fidelity: requested property vs. property of the generated molecule\n"
                 "(gain = shuffled-target MAE / actual MAE; ~1.0x means conditioning is being ignored)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_distribution_distances(dist_metrics, out_path):
    """
    Per-property real-vs-generated distance. Two panels because the scales are
    different beasts: normalized Wasserstein (in units of the real property's
    std, so MW and LogP are comparable) and KL(gen||real).
    """
    per_prop = dist_metrics.get("per_property", {})
    if not per_prop:
        return
    props = list(per_prop.keys())
    w = [per_prop[p]["wasserstein_norm"] for p in props]
    kl = [per_prop[p]["kl_gen_vs_real"] for p in props]
    x = np.arange(len(props))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x, w, color="#4C72B0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(props, rotation=30, ha="right")
    axes[0].set_ylabel("W1 / std(real)")
    axes[0].set_title("Normalized Wasserstein-1 distance (lower = better)")
    axes[0].axhline(0.1, color="green", linestyle="--", linewidth=1, label="0.1 std (close match)")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, kl, color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(props, rotation=30, ha="right")
    axes[1].set_ylabel("KL(generated || real)")
    title = "KL divergence per property (lower = better)"
    score = dist_metrics.get("guacamol_kl_score")
    if score is not None:
        title += f"\nGuacaMol KL score = mean exp(-KL) = {score:.3f}  (1.0 = perfect)"
    axes[1].set_title(title)

    fig.suptitle("Generated vs. real property-distribution distances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_diversity(snn_values, div_metrics, out_path):
    """
    Left: nearest-neighbour Tanimoto to the training set. A mass piled up near
    1.0 means the model is copying training chemistry with cosmetic edits -
    which the 'novelty' metric (exact string match) happily scores as 100%.
    Right: the headline diversity numbers, including macrocycle rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    snn = np.asarray(snn_values, dtype=float)
    if snn.size:
        axes[0].hist(snn, bins=40, color="#55A868", alpha=0.8)
        axes[0].axvline(float(snn.mean()), color="black", linestyle="--", linewidth=1,
                        label=f"mean={snn.mean():.2f}")
        axes[0].axvline(0.9, color="red", linestyle=":", linewidth=1.2,
                        label="0.9 = near-duplicate")
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "no data", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xlabel("max Tanimoto similarity to any training molecule (SNN)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Nearest-neighbour similarity to the training set")

    keys = [
        ("intdiv1", "IntDiv1"),
        ("intdiv2", "IntDiv2"),
        ("scaffold_uniqueness", "scaffold uniqueness"),
        ("scaffold_entropy_normalized", "scaffold entropy (norm)"),
        ("frag_similarity_cosine", "BRICS frag similarity"),
        ("macrocycle_rate", "macrocycle rate (ring>=12)"),
    ]
    labels = [lab for k, lab in keys if div_metrics.get(k) is not None]
    vals = [div_metrics[k] for k, _ in keys if div_metrics.get(k) is not None]
    y = np.arange(len(labels))
    colors = ["#8172B2"] * len(labels)
    if "macrocycle rate (ring>=12)" in labels:
        colors[labels.index("macrocycle rate (ring>=12)")] = "#C44E52"
    axes[1].barh(y, vals, color=colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlim(0, 1.05)
    for yi, v in zip(y, vals):
        axes[1].text(min(v + 0.02, 0.98), yi, f"{v:.2f}", va="center", fontsize=8)
    axes[1].set_title("Diversity & domain checks (all on 0-1 scale)")

    fig.suptitle("Sample diversity: is the model collapsing, or copying?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_optimization_scores(scores_before, scores_after, metric_name, out_path):
    """Compare a score distribution before vs after optimization (rejection
    sampling or RL fine-tuning) — e.g. QED before/after best-of-N filtering."""
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.histogram_bin_edges(np.concatenate([scores_before, scores_after]), bins=30)
    ax.hist(scores_before, bins=bins, density=True, alpha=0.5, label="before", color="#4C72B0")
    ax.hist(scores_after, bins=bins, density=True, alpha=0.5, label="after", color="#55A868")
    ax.axvline(np.mean(scores_before), color="#4C72B0", linestyle="--", linewidth=1)
    ax.axvline(np.mean(scores_after), color="#55A868", linestyle="--", linewidth=1)
    ax.set_xlabel(metric_name)
    ax.set_ylabel("density")
    ax.set_title(f"{metric_name}: before vs after optimization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
