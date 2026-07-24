"""
Full evaluation report for a trained checkpoint.

Generates molecules across all classes and scores them on four axes:

  1. MODEL QUALITY (likelihood)   - perplexity, bits/token, predictive Shannon
                                    entropy, next-token accuracy, train-vs-val
                                    perplexity gap, per-class perplexity.
  2. DISTRIBUTION MATCH           - per-property Wasserstein-1, KL, Jensen-
                                    Shannon, KS vs the real training set, plus
                                    the GuacaMol KL benchmark score.
  3. CONDITIONING FIDELITY        - did asking for MW=500 actually get MW=500,
                                    and did asking for a Macrolide get a
                                    Macrolide? Scored against shuffled-target
                                    and majority-class controls.
  4. SAMPLE QUALITY / DIVERSITY   - validity/uniqueness/novelty, IntDiv, SNN,
                                    scaffold entropy, fragment similarity,
                                    token entropy, truncation rate, and the
                                    macrocycle rate (are they even macrocycles?).

Run:
    python evaluate.py --data data/processed.pkl --ckpt checkpoints/model.pt \
        --n_per_class 200 --out_dir outputs/eval

    # everything, including the slow/optional bits
    python evaluate.py --data data/processed.pkl --ckpt checkpoints/model.pt \
        --n_per_class 500 --compute_fcd --out_dir outputs/eval

Produces (in --out_dir):
    metrics.json                 - every number below, nested by family
    property_distributions.png   - MW/LogP/PSA/RotB/HBA/HBD, generated vs real
    validity_by_class.png        - validity/uniqueness/novelty per class
    qed_sa_distributions.png     - QED and SA score of generated molecules
    conditioning_fidelity.png    - requested vs. achieved property scatter (NEW)
    distribution_distances.png   - per-property W1 and KL bars (NEW)
    diversity.png                - SNN histogram + diversity/domain checks (NEW)
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger

import metrics as M
from model import ConditionalSMILESTransformer
from chem_utils import mol_from_sequence, compute_properties, validity_uniqueness_novelty
from plotting_utils import (
    plot_property_distributions, plot_validity_by_class, plot_qed_sa,
    plot_conditioning_fidelity, plot_distribution_distances, plot_diversity,
)

RDLogger.DisableLog("rdApp.*")


def generate_for_class(model, data, cls, n, device, temperature=1.0, top_k=30, top_p=0.95):
    """
    Generate n molecules for a class, conditioned on that class's real property
    distribution (sampling real rows' property vectors, not just the class
    mean) so the eval reflects realistic within-class diversity.

    Returns (sequences, cond_rows). The conditioning rows are returned - not
    just the strings - because conditioning fidelity needs to know exactly what
    was ASKED for on a per-molecule basis, to compare against what came out.
    """
    n_classes = len(data["class_list"])
    cls_idx = data["class_to_idx"][cls]
    all_class_idx = np.argmax(data["cond"][:, :n_classes], axis=1)
    pool = np.where(all_class_idx == cls_idx)[0]
    # Always draw exactly n conditioning vectors, resampling WITH replacement when
    # the class pool is smaller than n. The old code used size=min(n, len(pool)),
    # which silently generated fewer molecules for rare classes (Macrocyclic
    # Polyene has ~81 members, so --n_per_class 200 quietly became 81) - i.e. the
    # noisiest per-class estimates were exactly the ones with the fewest samples.
    chosen = np.random.choice(pool, size=n, replace=len(pool) < n)
    cond_rows = data["cond"][chosen]
    cond = torch.tensor(cond_rows, dtype=torch.float32, device=device)
    seqs = model.generate(
        cond, data["stoi"], data["itos"], max_new_tokens=data["max_len"],
        temperature=temperature, top_k=top_k, top_p=top_p, device=device,
    )
    return seqs, cond_rows


def denormalize_targets(cond_rows, data):
    """
    Conditioning vector -> the raw property values it encodes.

    cond = [class one-hot | z-scored properties], so undo the z-scoring with the
    same mean/std data_prep.py saved. This recovers the human-readable target
    ("we asked for MW=512.3") that the fidelity metric compares against.
    """
    n_classes = len(data["class_list"])
    props = data["cond_properties"]
    z = cond_rows[:, n_classes:]
    raw = np.zeros_like(z, dtype=float)
    for j, col in enumerate(props):
        s = data["prop_stats"][col]
        raw[:, j] = z[:, j] * s["std"] + s["mean"]
    return pd.DataFrame(raw, columns=props)


def sample_real_properties(data, n_sample, seed=0):
    """
    Compute FULL RDKit properties (incl. QED and SA) for a random subsample of
    real training molecules.

    The old evaluate.py punted here - it wrote a `real_QED_note` saying "not
    precomputed, compare qualitatively". But QED/SA are exactly the properties
    you most want a real baseline for: "generated QED = 0.42" is meaningless
    without knowing the training set sits at 0.38 (fine) or 0.75 (bad).
    """
    rng = np.random.default_rng(seed)
    smiles = data["smiles"]
    idx = rng.choice(len(smiles), size=min(n_sample, len(smiles)), replace=False)
    rows = []
    for i in idx:
        mol = Chem.MolFromSmiles(smiles[i])
        if mol is not None:
            rows.append(compute_properties(mol))
    return pd.DataFrame(rows)


def try_compute_fcd(real_smiles, gen_smiles):
    """
    Frechet ChemNet Distance: how close the generated distribution is to the
    real one in a pretrained ChemNet embedding space (lower = more realistic).
    Optional: needs `pip install fcd` plus a one-time model download.
    """
    try:
        import fcd
        model = fcd.load_ref_model()
        real_act = fcd.get_predictions(model, real_smiles)
        gen_act = fcd.get_predictions(model, gen_smiles)
        mu1, sigma1 = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
        mu2, sigma2 = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)
        return float(fcd.calculate_frechet_distance(mu1, sigma1, mu2, sigma2))
    except ImportError:
        print("[evaluate] `fcd` not installed - skipping FCD. `pip install fcd` to enable.")
        return None
    except Exception as e:
        print(f"[evaluate] FCD computation failed ({e}) - skipping.")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed.pkl")
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--n_per_class", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_real_sample", type=int, default=2000,
                    help="real molecules to RDKit-profile for the QED/SA/property baselines")
    ap.add_argument("--ppl_train_sample", type=int, default=5000,
                    help="train molecules scored for the train-vs-val perplexity gap (0 = skip)")
    ap.add_argument("--compute_fcd", action="store_true",
                    help="also compute Frechet ChemNet Distance (needs `pip install fcd`, internet)")
    ap.add_argument("--skip_class_fidelity", action="store_true",
                    help="skip the RandomForest class-fidelity oracle (the slowest metric)")
    ap.add_argument("--skip_frag", action="store_true",
                    help="skip BRICS fragment similarity (slow on large macrocycles)")
    ap.add_argument("--out_dir", default="outputs/eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    representation = data.get("representation", "smiles")
    cond_props = data["cond_properties"]

    ckpt = torch.load(args.ckpt, map_location=device)
    model = ConditionalSMILESTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_smiles_set = set(data["smiles"])
    real_df = data["real_properties"]  # MW/LogP/HBA/HBD/PSA/RotB/M_type for every training molecule

    # =====================================================================
    # 1. MODEL QUALITY - perplexity / entropy (needs the model, not samples)
    # =====================================================================
    print("[1/5] Likelihood metrics (perplexity, entropy) ...")
    likelihood = M.model_likelihood_metrics(model, data, data["val_idx"], device, tag="val")
    if args.ppl_train_sample > 0:
        tr = data["train_idx"]
        sub = np.random.choice(tr, size=min(args.ppl_train_sample, len(tr)), replace=False)
        likelihood.update(M.model_likelihood_metrics(model, data, sub, device, tag="train"))
        likelihood["train_val_perplexity_gap"] = (
            likelihood["val_perplexity"] - likelihood["train_perplexity"]
        )
    print(f"  val perplexity      = {likelihood['val_perplexity']:.3f}  "
          f"({likelihood['val_bits_per_token']:.3f} bits/token)")
    print(f"  predictive entropy  = {likelihood['val_predictive_entropy_nats']:.3f} nats/token")
    print(f"  next-token accuracy = {likelihood['val_next_token_accuracy']:.2%}")
    if "train_val_perplexity_gap" in likelihood:
        print(f"  train->val ppl gap  = {likelihood['train_val_perplexity_gap']:+.3f} "
              f"(large positive = overfitting)")

    # =====================================================================
    # 2. GENERATE, tracking the conditioning target for every sample
    # =====================================================================
    print(f"\n[2/5] Generating {args.n_per_class} molecules per class "
          f"({len(data['class_list'])} classes) ...")
    all_gen_rows, all_target_rows, all_seqs = [], [], []
    per_class_metrics = {}

    for cls in data["class_list"]:
        seqs, cond_rows = generate_for_class(
            model, data, cls, args.n_per_class, device,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        targets = denormalize_targets(cond_rows, data)
        all_seqs.extend(seqs)

        valid_smiles, validity, uniqueness, novelty = validity_uniqueness_novelty(
            seqs, representation, train_smiles_set)
        per_class_metrics[cls] = {
            "n_generated": len(seqs), "validity": validity,
            "uniqueness": uniqueness, "novelty": novelty,
        }
        print(f"  {cls:22s} validity={validity:.2%}  uniqueness={uniqueness:.2%}  novelty={novelty:.2%}")

        for i, seq in enumerate(seqs):
            mol = mol_from_sequence(seq, representation)
            if mol is None:
                continue
            row = compute_properties(mol)
            row["M_type"] = cls
            row["canonical_smiles"] = Chem.MolToSmiles(mol)
            all_gen_rows.append(row)
            # target row is kept ONLY for valid molecules, so the two frames
            # stay index-aligned for the fidelity comparison
            t = targets.iloc[i].to_dict()
            t["M_type"] = cls
            all_target_rows.append(t)

    gen_df = pd.DataFrame(all_gen_rows)
    target_df = pd.DataFrame(all_target_rows)

    overall_validity = float(np.mean([m["validity"] for m in per_class_metrics.values()]))
    overall_uniqueness = float(np.mean([m["uniqueness"] for m in per_class_metrics.values()]))
    overall_novelty = float(np.mean([m["novelty"] for m in per_class_metrics.values()]))
    print(f"\n  macro-avg: validity={overall_validity:.2%}  "
          f"uniqueness={overall_uniqueness:.2%}  novelty={overall_novelty:.2%}")

    if gen_df.empty:
        print("\nNo valid molecules generated - skipping downstream metrics. "
              "Check the checkpoint and the training run.")
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump({"per_class": per_class_metrics, "overall_validity": overall_validity}, f, indent=2)
        return

    # =====================================================================
    # 3. CONDITIONING FIDELITY  <- the headline new metric
    # =====================================================================
    print("\n[3/5] Conditioning fidelity ...")
    fidelity = M.property_conditioning_fidelity(
        target_df, gen_df, data["prop_stats"], cond_props, seed=args.seed)
    for prop, m in fidelity["per_property"].items():
        print(f"  {prop:5s} MAE={m['mae']:8.2f}  r={m['pearson_r']:+.3f}  "
              f"R2={m['r2']:+.3f}  shuffled-MAE={m['mae_shuffled_control']:8.2f}  "
              f"gain={m['fidelity_gain']:.2f}x")
    print(f"  mean fidelity gain = {fidelity['mean_fidelity_gain']:.2f}x  "
          f"(~1.0 = conditioning ignored; >1.5 = steering works)")

    class_fidelity = None
    if not args.skip_class_fidelity:
        print("  training fingerprint oracle for class fidelity ...")
        class_fidelity = M.class_conditioning_fidelity(
            data["smiles"], real_df["M_type"].tolist(),
            gen_df["canonical_smiles"].tolist(), gen_df["M_type"].tolist(),
            data["class_list"], seed=args.seed)
        if class_fidelity:
            print(f"  class fidelity = {class_fidelity['class_fidelity_accuracy']:.2%} "
                  f"(balanced {class_fidelity['class_fidelity_balanced_accuracy']:.2%})  |  "
                  f"oracle ceiling = {class_fidelity['oracle_holdout_accuracy']:.2%}  |  "
                  f"majority baseline = {class_fidelity['majority_class_baseline']:.2%}")

    # =====================================================================
    # 4. DISTRIBUTION MATCH (incl. QED/SA vs a real baseline)
    # =====================================================================
    print("\n[4/5] Distribution distances (Wasserstein / KL / JS) ...")
    real_full_df = sample_real_properties(data, args.n_real_sample, seed=args.seed)
    dist = M.distribution_distances(real_full_df, gen_df)
    for prop, m in dist["per_property"].items():
        print(f"  {prop:5s} W1={m['wasserstein']:8.3f}  W1/std={m['wasserstein_norm']:.3f}  "
              f"KL={m['kl_gen_vs_real']:.4f}  JS={m['js_distance']:.4f}")
    print(f"  GuacaMol KL score = {dist['guacamol_kl_score']:.3f} (1.0 = perfect match)")
    per_class_dist = M.per_class_distribution_distances(real_df, gen_df, data["class_list"])

    # =====================================================================
    # 5. DIVERSITY / SAMPLE QUALITY
    # =====================================================================
    print("\n[5/5] Diversity, entropy, and domain checks ...")
    diversity = M.diversity_metrics(
        gen_df["canonical_smiles"].tolist(), data["smiles"],
        seed=args.seed, compute_frag=not args.skip_frag)
    snn_values = diversity.pop("_snn_values", [])
    diversity["macrocycle_rate_real_reference"] = M.macrocycle_rate_real(data["smiles"], seed=args.seed)

    seq_entropy = M.sequence_entropy_metrics(all_seqs, data["sequences"], representation)
    seq_entropy["truncation_rate"] = M.truncation_rate(all_seqs, representation, data["max_len"])

    print(f"  IntDiv1={diversity['intdiv1']:.3f}  SNN-to-train={diversity['snn_to_train_mean']:.3f}  "
          f"scaffold-uniqueness={diversity['scaffold_uniqueness']:.3f}")
    print(f"  macrocycle rate = {diversity['macrocycle_rate']:.2%}  "
          f"(real reference = {diversity['macrocycle_rate_real_reference']:.2%})")
    print(f"  token entropy: gen={seq_entropy['gen_token_entropy_nats']:.3f} vs "
          f"real={seq_entropy['real_token_entropy_nats']:.3f} nats  "
          f"(truncation rate = {seq_entropy['truncation_rate']:.2%})")

    # =====================================================================
    # Assemble + save
    # =====================================================================
    metrics = {
        "representation": representation,
        "n_generated_total": int(len(all_seqs)),
        "n_valid_total": int(len(gen_df)),
        "standard": {
            "overall_validity": overall_validity,
            "overall_uniqueness": overall_uniqueness,
            "overall_novelty": overall_novelty,
            "per_class": per_class_metrics,
        },
        "model_quality": likelihood,
        "conditioning_fidelity": {
            "property": fidelity,
            "class": class_fidelity,
        },
        "distribution_match": {
            "overall": dist,
            "per_class": per_class_dist,
        },
        "diversity": diversity,
        "sequence_entropy": seq_entropy,
        "generated_property_means": {
            c: float(gen_df[c].mean()) for c in ["MW", "LogP", "QED", "SA", "num_rings"]
            if c in gen_df.columns and gen_df[c].notna().any()
        },
        "real_property_means": {
            c: float(real_full_df[c].mean()) for c in ["MW", "LogP", "QED", "SA", "num_rings"]
            if c in real_full_df.columns and real_full_df[c].notna().any()
        },
    }

    if args.compute_fcd:
        idx = np.random.choice(len(data["smiles"]), size=min(2000, len(data["smiles"])), replace=False)
        metrics["distribution_match"]["fcd"] = try_compute_fcd(
            [data["smiles"][i] for i in idx], gen_df["canonical_smiles"].tolist())

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved metrics -> {os.path.join(args.out_dir, 'metrics.json')}")

    print("Plotting ...")
    plot_property_distributions(real_df, gen_df, os.path.join(args.out_dir, "property_distributions.png"))
    plot_validity_by_class(per_class_metrics, os.path.join(args.out_dir, "validity_by_class.png"))
    plot_qed_sa(gen_df, os.path.join(args.out_dir, "qed_sa_distributions.png"))
    plot_conditioning_fidelity(target_df, gen_df, fidelity, cond_props,
                               os.path.join(args.out_dir, "conditioning_fidelity.png"))
    plot_distribution_distances(dist, os.path.join(args.out_dir, "distribution_distances.png"))
    plot_diversity(snn_values, diversity, os.path.join(args.out_dir, "diversity.png"))
    print(f"Saved 6 plots -> {args.out_dir}/*.png")


if __name__ == "__main__":
    main()
