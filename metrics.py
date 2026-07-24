"""
Quantitative evaluation metrics for the conditional macrocycle generator.

Four families, answering four different questions:

  1. MODEL-QUALITY (likelihood-based) - "how well does the model fit the data?"
     perplexity / NLL-per-token / bits-per-token, predictive Shannon entropy,
     next-token top-1 accuracy, and the train-vs-val perplexity gap (overfitting).
     These need the model itself, not just its samples.

  2. DISTRIBUTION-MATCH - "do generated molecules look like real ones?"
     Per-property Wasserstein-1, KL(gen||real), Jensen-Shannon distance, and
     KS statistic, for MW/LogP/HBA/HBD/PSA/RotB/QED/SA. Wasserstein is also
     reported in units of the real property's std ("normalized W1") so the six
     properties are comparable to each other on one axis. The mean of
     exp(-KL) across properties is GuacaMol's KL-divergence benchmark score
     (1.0 = perfect match).

  3. CONDITIONING FIDELITY - "does the conditioning vector actually steer it?"
     This is the metric the project was missing entirely. Validity/novelty say
     nothing about whether asking for MW=500 gets you MW=500. Two parts:
       (a) property fidelity: MAE / RMSE / R^2 / Pearson-r between the property
           we ASKED for and the property RDKit measures on what came out. Each
           is reported against a SHUFFLED-CONDITION CONTROL: the same generated
           molecules re-paired with someone else's target. A model that ignores
           conditioning entirely still scores a decent MAE (it emits typical
           molecules, and typical targets are typical), so the raw MAE alone is
           not evidence of steering. The ratio shuffled_MAE / actual_MAE is:
           ~1.0 = conditioning is being ignored; >>1.0 = it is being followed.
       (b) class fidelity: a Morgan-fingerprint RandomForest oracle trained on
           the REAL molecules predicts M_type for each generated molecule; we
           report how often it agrees with the class we conditioned on. The
           oracle's own held-out accuracy is reported alongside, since it puts
           a ceiling on how much the generated-set number can be trusted.

  4. SAMPLE QUALITY / DIVERSITY - "is it just memorizing or collapsing?"
     IntDiv1/IntDiv2 (Tanimoto), SNN (nearest-neighbour similarity to the
     training set), scaffold uniqueness + scaffold Shannon entropy (mode
     collapse detector), BRICS fragment cosine similarity, generated-token
     unigram entropy + KL against the real corpus, and a domain-specific check
     the generic metrics all miss: MACROCYCLE RATE, the fraction of generated
     molecules that actually contain a >=12-membered ring. A model can score
     100% validity while emitting perfectly valid non-macrocycles, which for
     this dataset is a silent failure.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Property groups. HBA/HBD/RotB are integer counts, so their KL is computed
# from the exact PMF over the observed support rather than from a KDE: it is
# estimator-free, with no bandwidth to tune. It is NOT the lower-variance
# choice - see the note in distribution_distances(), where the measured noise
# floors are reported instead of assumed away.
# ---------------------------------------------------------------------------
CONTINUOUS_PROPS = ["MW", "LogP", "PSA", "QED", "SA"]
DISCRETE_PROPS = ["HBA", "HBD", "RotB"]
EPS = 1e-10


# ===========================================================================
# 1. MODEL-QUALITY METRICS (need torch + the model)
# ===========================================================================
def model_likelihood_metrics(model, data, indices, device, batch_size=128, tag="val"):
    """
    Teacher-forced pass over `indices`, accumulating token-level statistics.

    Returns perplexity, NLL/token (nats), bits/token, mean predictive Shannon
    entropy, and top-1 next-token accuracy - plus a per-class perplexity
    breakdown, which is the likelihood-side analogue of the per-class validity
    report: overall perplexity can look fine while a rare class is modelled
    badly.

    NOTE ON CORRECTNESS: sums are accumulated over NON-PAD TARGET TOKENS and
    divided once at the end. Averaging per-batch mean losses instead (the
    common shortcut) silently weights short sequences more heavily, because
    each batch has a different number of real tokens.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    pad_idx = data["stoi"]["<pad>"]
    n_classes = len(data["class_list"])

    token_ids = data["token_ids"][indices]
    cond = data["cond"][indices]
    class_idx = np.argmax(cond[:, :n_classes], axis=1)

    tot_nll = tot_ent = 0.0
    tot_tokens = tot_correct = 0
    cls_nll = np.zeros(n_classes)
    cls_tokens = np.zeros(n_classes)

    with torch.no_grad():
        for start in range(0, len(token_ids), batch_size):
            ids = torch.from_numpy(token_ids[start:start + batch_size]).long().to(device)
            c = torch.from_numpy(cond[start:start + batch_size]).float().to(device)
            x, y = ids[:, :-1], ids[:, 1:]          # same shift as SmilesDataset

            logits, _ = model(x, c)                 # (B, T, V)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()

            # per-token NLL, zeroed at pad positions by ignore_index
            nll = F.nll_loss(
                log_probs.reshape(-1, log_probs.size(-1)),
                y.reshape(-1),
                ignore_index=pad_idx,
                reduction="none",
            ).view(y.shape)                         # (B, T)

            ent = -(probs * log_probs).sum(-1)      # (B, T) predictive entropy
            mask = (y != pad_idx)
            correct = (log_probs.argmax(-1) == y) & mask

            tot_nll += nll[mask].sum().item()
            tot_ent += ent[mask].sum().item()
            tot_correct += correct.sum().item()
            tot_tokens += mask.sum().item()

            row_nll = nll.sum(dim=1).cpu().numpy()
            row_tok = mask.sum(dim=1).cpu().numpy()
            batch_cls = class_idx[start:start + batch_size]
            for ci in range(n_classes):
                sel = batch_cls == ci
                if sel.any():
                    cls_nll[ci] += row_nll[sel].sum()
                    cls_tokens[ci] += row_tok[sel].sum()

    nll_per_token = tot_nll / max(1, tot_tokens)
    per_class_ppl = {
        data["class_list"][ci]: (float(np.exp(cls_nll[ci] / cls_tokens[ci]))
                                 if cls_tokens[ci] > 0 else None)
        for ci in range(n_classes)
    }

    return {
        f"{tag}_nll_per_token": float(nll_per_token),
        f"{tag}_perplexity": float(np.exp(nll_per_token)),
        f"{tag}_bits_per_token": float(nll_per_token / math.log(2)),
        f"{tag}_predictive_entropy_nats": float(tot_ent / max(1, tot_tokens)),
        f"{tag}_next_token_accuracy": float(tot_correct / max(1, tot_tokens)),
        f"{tag}_n_tokens_scored": int(tot_tokens),
        f"{tag}_per_class_perplexity": per_class_ppl,
    }


# ===========================================================================
# 2. SEQUENCE / TOKEN ENTROPY (no model needed - strings only)
# ===========================================================================
def _tokenize(seq, representation):
    if representation == "selfies":
        import selfies as sf
        try:
            return list(sf.split_selfies(seq))
        except Exception:
            return []
    return list(seq)


def _pmf_from_counter(counter, support):
    total = sum(counter.values())
    if total == 0:
        return np.full(len(support), 1.0 / max(1, len(support)))
    return np.array([counter.get(s, 0) / total for s in support])


def sequence_entropy_metrics(gen_seqs, real_seqs, representation):
    """
    Unigram token statistics of the GENERATED corpus vs the REAL one.

    - shannon entropy (nats) of the token distribution, and its normalized
      form H / log(V), which is 1.0 for a uniform distribution over the
      vocabulary actually used and -> 0 as the model collapses onto a few
      tokens. A generated entropy well BELOW the real one is a mode-collapse
      / degenerate-repetition signature that validity does not catch.
    - KL(gen || real) and Jensen-Shannon distance over the token unigrams.
    """
    from scipy.spatial.distance import jensenshannon

    gen_tokens = [t for s in gen_seqs for t in _tokenize(s, representation)]
    real_tokens = [t for s in real_seqs for t in _tokenize(s, representation)]
    if not gen_tokens or not real_tokens:
        return {}

    gen_c, real_c = Counter(gen_tokens), Counter(real_tokens)
    support = sorted(set(gen_c) | set(real_c))
    p = _pmf_from_counter(gen_c, support) + EPS
    q = _pmf_from_counter(real_c, support) + EPS
    p /= p.sum()
    q /= q.sum()

    def _H(pmf):
        pmf = pmf[pmf > 0]
        return float(-(pmf * np.log(pmf)).sum())

    gen_lengths = np.array([len(_tokenize(s, representation)) for s in gen_seqs])
    real_lengths = np.array([len(_tokenize(s, representation)) for s in real_seqs])

    return {
        "gen_token_entropy_nats": _H(p),
        "real_token_entropy_nats": _H(q),
        "gen_token_entropy_normalized": _H(p) / math.log(len(support)) if len(support) > 1 else 0.0,
        "real_token_entropy_normalized": _H(q) / math.log(len(support)) if len(support) > 1 else 0.0,
        "token_unigram_kl_gen_vs_real": float((p * np.log(p / q)).sum()),
        "token_unigram_js_distance": float(jensenshannon(p, q, base=np.e)),
        "gen_mean_length": float(gen_lengths.mean()),
        "real_mean_length": float(real_lengths.mean()),
        "gen_unique_tokens_used": int(len(gen_c)),
        "real_unique_tokens_used": int(len(real_c)),
    }


def truncation_rate(gen_seqs, representation, max_len):
    """
    Fraction of samples that ran to the length cap without emitting <eos>.
    These get cut mid-molecule and are near-guaranteed invalid, so a high rate
    means "raise max_len / fix EOS modelling", not "the chemistry is wrong" -
    a distinction the bare validity number hides.
    """
    cap = max_len - 2  # BOS + tokens + EOS budget
    n_trunc = sum(1 for s in gen_seqs if len(_tokenize(s, representation)) >= cap)
    return float(n_trunc / len(gen_seqs)) if gen_seqs else 0.0


# ===========================================================================
# 3. DISTRIBUTION-MATCH METRICS (per property)
# ===========================================================================
def _kl_js_continuous(real, gen, n_grid=200):
    """KDE both samples on a shared grid, then discrete KL/JS. GuacaMol-style."""
    from scipy.stats import gaussian_kde
    from scipy.spatial.distance import jensenshannon

    lo = min(real.min(), gen.min())
    hi = max(real.max(), gen.max())
    if hi - lo < 1e-9:
        return 0.0, 0.0
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    try:
        q = gaussian_kde(real)(grid)               # real
        p = gaussian_kde(gen)(grid)                # generated
    except Exception:  # singular covariance (e.g. constant sample) -> histogram
        bins = np.linspace(lo - pad, hi + pad, 51)
        q, _ = np.histogram(real, bins=bins, density=True)
        p, _ = np.histogram(gen, bins=bins, density=True)
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    p /= p.sum()
    q /= q.sum()
    return float((p * np.log(p / q)).sum()), float(jensenshannon(p, q, base=np.e))


def _kl_js_discrete(real, gen):
    """Integer-valued properties: exact PMF over the union of observed values."""
    from scipy.spatial.distance import jensenshannon

    real_i = np.rint(real).astype(int)
    gen_i = np.rint(gen).astype(int)
    support = sorted(set(real_i.tolist()) | set(gen_i.tolist()))
    q = _pmf_from_counter(Counter(real_i.tolist()), support) + EPS
    p = _pmf_from_counter(Counter(gen_i.tolist()), support) + EPS
    p /= p.sum()
    q /= q.sum()
    return float((p * np.log(p / q)).sum()), float(jensenshannon(p, q, base=np.e))


def _distances_once(real, gen, prop, discrete):
    """One property, one pair of samples -> the four distances."""
    from scipy.stats import wasserstein_distance, ks_2samp

    w1 = float(wasserstein_distance(real, gen))
    std = float(real.std()) or 1.0
    kl, js = _kl_js_discrete(real, gen) if discrete else _kl_js_continuous(real, gen)
    ks = ks_2samp(real, gen)
    return {
        "wasserstein": w1,
        "wasserstein_norm": float(w1 / std),
        "kl_gen_vs_real": kl,
        "js_distance": js,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def distribution_distances(real_df, gen_df, properties=None, noise_floor=True, seed=0):
    """
    Per-property real-vs-generated distances.

    wasserstein          - W1 in the property's own units (g/mol, logP units...)
    wasserstein_norm     - W1 / std(real); comparable ACROSS properties, which
                           raw W1 is not (MW spans hundreds, LogP spans ~10)
    kl_gen_vs_real       - KL(P_gen || P_real); 0 = identical
    js_distance          - symmetric, bounded [0, sqrt(ln 2)]; robust when the
                           supports only partially overlap (KL blows up there)
    ks_statistic / p     - two-sample Kolmogorov-Smirnov
    Aggregate: guacamol_kl_score = mean(exp(-KL)) in [0, 1], 1.0 = perfect.

    ESTIMATORS. Continuous properties use a KDE on a shared grid; the integer
    ones (HBA/HBD/RotB) use the exact PMF over the observed support, which is
    estimator-free (no bandwidth to choose). Note the PMF is NOISIER, not
    cleaner: measured on two identical Poisson(10) samples at n=2000, the PMF
    reports KL ~= 0.023 and a KDE ~= 0.006, so the KDE actually has the better
    signal-to-noise ratio. Both estimators rank models identically once the
    noise floor below is subtracted, so we keep the estimator-free one and
    MEASURE its floor rather than pretending it is zero. `kl_alt_kde` is
    reported alongside for the discrete properties so this is auditable.

    NOISE FLOOR (`real_vs_real_*`). None of these distances is zero for a
    perfect model at finite n - two halves of the REAL dataset already differ
    by sampling noise. So we split the real sample in half and measure the
    distance between the halves. That is the value a PERFECT generator would
    score. Read `kl_excess` (= kl - floor, clipped at 0) and
    `above_noise_floor`: a KL of 0.02 on HBA is not a modelling failure, it is
    what n=2000 looks like.
    """
    if properties is None:
        properties = CONTINUOUS_PROPS + DISCRETE_PROPS
    rng = np.random.default_rng(seed)

    out = {}
    kls = []
    for prop in properties:
        if prop not in real_df.columns or prop not in gen_df.columns:
            continue
        real = pd.to_numeric(real_df[prop], errors="coerce").dropna().values.astype(float)
        gen = pd.to_numeric(gen_df[prop], errors="coerce").dropna().values.astype(float)
        if len(real) < 2 or len(gen) < 2:
            continue
        discrete = prop in DISCRETE_PROPS

        d = _distances_once(real, gen, prop, discrete)
        d.update({
            "real_mean": float(real.mean()), "gen_mean": float(gen.mean()),
            "real_std": float(real.std()), "gen_std": float(gen.std()),
            "n_real": int(len(real)), "n_gen": int(len(gen)),
        })
        if discrete:
            d["kl_alt_kde"] = _kl_js_continuous(real, gen)[0]

        # --- what would a PERFECT generator score at this sample size? ------
        if noise_floor and len(real) >= 20:
            perm = rng.permutation(len(real))
            h1, h2 = real[perm[: len(real) // 2]], real[perm[len(real) // 2:]]
            floor = _distances_once(h1, h2, prop, discrete)
            d["real_vs_real_kl_floor"] = floor["kl_gen_vs_real"]
            d["real_vs_real_w1_norm_floor"] = floor["wasserstein_norm"]
            d["real_vs_real_js_floor"] = floor["js_distance"]
            d["kl_excess"] = float(max(0.0, d["kl_gen_vs_real"] - floor["kl_gen_vs_real"]))
            d["above_noise_floor"] = bool(d["kl_gen_vs_real"] > 2 * floor["kl_gen_vs_real"])

        out[prop] = d
        kls.append(d["kl_gen_vs_real"])

    return {
        "per_property": out,
        "guacamol_kl_score": float(np.mean([math.exp(-k) for k in kls])) if kls else None,
        "mean_wasserstein_norm": float(np.mean([v["wasserstein_norm"] for v in out.values()])) if out else None,
        "mean_js_distance": float(np.mean([v["js_distance"] for v in out.values()])) if out else None,
        "note": ("compare each kl_gen_vs_real against its real_vs_real_kl_floor; "
                 "distances below ~2x the floor are sampling noise, not modelling error"),
    }


def per_class_distribution_distances(real_df, gen_df, class_list, properties=("MW", "LogP")):
    """Same idea, sliced by M_type - catches a class-specific distribution miss
    that the pooled numbers average away."""
    out = {}
    for cls in class_list:
        r = real_df[real_df["M_type"] == cls]
        g = gen_df[gen_df["M_type"] == cls]
        if len(r) < 5 or len(g) < 5:
            out[cls] = None
            continue
        out[cls] = distribution_distances(r, g, list(properties))["per_property"]
    return out


# ===========================================================================
# 4. CONDITIONING FIDELITY  <- the metric the project was missing
# ===========================================================================
def property_conditioning_fidelity(target_df, achieved_df, prop_stats, properties, seed=0):
    """
    For every generated molecule we know the exact property vector we asked for
    (the conditioning vector, de-normalized back to raw units). RDKit tells us
    what we actually got. This scores the gap.

    SHUFFLED CONTROL (important): a model that completely ignores conditioning
    still gets a respectable MAE, because it emits distribution-typical
    molecules and the targets are distribution-typical too. So every metric is
    also computed against randomly re-paired targets. Read:

        fidelity_gain = shuffled_MAE / MAE
          ~1.0  -> conditioning ignored (the number was an illusion)
          >1.5  -> conditioning is genuinely steering generation
    """
    rng = np.random.default_rng(seed)
    from scipy.stats import pearsonr, spearmanr

    out = {}
    for prop in properties:
        if prop not in target_df.columns or prop not in achieved_df.columns:
            continue
        t = pd.to_numeric(target_df[prop], errors="coerce").values.astype(float)
        a = pd.to_numeric(achieved_df[prop], errors="coerce").values.astype(float)
        ok = ~(np.isnan(t) | np.isnan(a))
        t, a = t[ok], a[ok]
        if len(t) < 5:
            continue

        err = np.abs(a - t)
        mae = float(err.mean())
        rmse = float(np.sqrt(((a - t) ** 2).mean()))

        # R^2 of the target as a predictor of the achieved value
        ss_res = float(((a - t) ** 2).sum())
        ss_tot = float(((a - a.mean()) ** 2).sum())
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        try:
            pear = float(pearsonr(t, a)[0])
            spear = float(spearmanr(t, a)[0])
        except Exception:
            pear = spear = float("nan")

        t_shuf = rng.permutation(t)
        mae_shuf = float(np.abs(a - t_shuf).mean())

        std = prop_stats[prop]["std"] if prop in prop_stats else (a.std() or 1.0)
        out[prop] = {
            "mae": mae,
            "rmse": rmse,
            "mae_in_std_units": float(mae / std),      # comparable across properties
            "r2": r2,
            "pearson_r": pear,
            "spearman_r": spear,
            "mae_shuffled_control": mae_shuf,
            "fidelity_gain": float(mae_shuf / mae) if mae > 0 else None,
            "mean_signed_bias": float((a - t).mean()),  # + = model overshoots the ask
            "n": int(len(t)),
        }

    gains = [v["fidelity_gain"] for v in out.values() if v["fidelity_gain"] is not None]
    pears = [v["pearson_r"] for v in out.values() if not math.isnan(v["pearson_r"])]
    return {
        "per_property": out,
        "mean_mae_in_std_units": float(np.mean([v["mae_in_std_units"] for v in out.values()])) if out else None,
        "mean_pearson_r": float(np.mean(pears)) if pears else None,
        "mean_fidelity_gain": float(np.mean(gains)) if gains else None,
        "interpretation": ("fidelity_gain ~1.0 means the conditioning vector is being ignored; "
                           ">1.5 means it is steering generation"),
    }


def class_conditioning_fidelity(real_smiles, real_labels, gen_smiles, gen_labels,
                                class_list, seed=0, max_real=8000, n_estimators=200):
    """
    Class-conditioning fidelity via a fingerprint oracle.

    Train a RandomForest on Morgan fingerprints of the REAL molecules to
    predict M_type, then ask it what class each GENERATED molecule looks like.
    Agreement with the class we conditioned on = class fidelity.

    Two guard-rails, both reported:
      - oracle_holdout_accuracy: the classifier's own accuracy on held-out REAL
        molecules. It is a ceiling on how meaningful the generated-set number
        is. If the oracle can only hit 70% on real data, an 65% score on
        generated molecules is not a damning result.
      - majority_class_baseline: what you'd get by always guessing the most
        common class. Beat this or the metric is vacuous.
    class_weight='balanced' is used because M_type is imbalanced ~331x.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    from chem_utils import morgan_fp_array

    rng = np.random.default_rng(seed)
    real_smiles = list(real_smiles)
    real_labels = list(real_labels)
    if len(real_smiles) > max_real:
        sel = rng.choice(len(real_smiles), size=max_real, replace=False)
        real_smiles = [real_smiles[i] for i in sel]
        real_labels = [real_labels[i] for i in sel]

    X_real, keep_real = morgan_fp_array(real_smiles)
    y_real = np.array([real_labels[i] for i in keep_real])
    X_gen, keep_gen = morgan_fp_array(gen_smiles)
    y_gen = np.array([gen_labels[i] for i in keep_gen])
    if len(X_real) < 50 or len(X_gen) < 10:
        return None

    # stratify only where every class has >=2 members, else fall back
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X_real, y_real, test_size=0.2, random_state=seed, stratify=y_real)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X_real, y_real, test_size=0.2, random_state=seed)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    ).fit(Xtr, ytr)

    oracle_acc = float(accuracy_score(yte, clf.predict(Xte)))
    oracle_bal = float(balanced_accuracy_score(yte, clf.predict(Xte)))

    pred_gen = clf.predict(X_gen)
    acc = float(accuracy_score(y_gen, pred_gen))
    bal = float(balanced_accuracy_score(y_gen, pred_gen))

    per_class = {}
    for cls in class_list:
        m = y_gen == cls
        per_class[cls] = float((pred_gen[m] == cls).mean()) if m.any() else None

    counts = Counter(real_labels)
    baseline = max(counts.values()) / sum(counts.values())
    labels_present = [c for c in class_list if c in set(y_gen) | set(pred_gen)]
    cm = confusion_matrix(y_gen, pred_gen, labels=labels_present)

    return {
        "class_fidelity_accuracy": acc,
        "class_fidelity_balanced_accuracy": bal,
        "per_class_fidelity": per_class,
        "oracle_holdout_accuracy": oracle_acc,
        "oracle_holdout_balanced_accuracy": oracle_bal,
        "majority_class_baseline": float(baseline),
        "confusion_matrix": {"labels": labels_present, "matrix": cm.tolist()},
        "n_generated_scored": int(len(y_gen)),
    }


# ===========================================================================
# 5. DIVERSITY / SAMPLE-QUALITY METRICS
# ===========================================================================
def diversity_metrics(gen_smiles, train_smiles, seed=0, sample_size=1000,
                      train_ref_size=5000, compute_frag=True):
    """
    IntDiv1 / IntDiv2  - 1 - mean (resp. root-mean-square) pairwise Tanimoto
                         within the generated set. Low = mode collapse.
    SNN                - mean nearest-neighbour Tanimoto from each generated
                         molecule to the training set. High (>~0.9) = the model
                         is regurgitating training chemistry with cosmetic
                         edits, even if 'novelty' says 100% (novelty is exact
                         string match, and one atom's difference defeats it).
    scaffold_uniqueness / scaffold_entropy - Bemis-Murcko scaffold diversity.
                         Normalized entropy -> 0 means everything shares a few
                         scaffolds; the sharpest mode-collapse detector here.
    frag_similarity    - cosine similarity of BRICS fragment frequency vectors,
                         generated vs real (MOSES 'Frag'). 1.0 = same building
                         blocks in the same proportions.
    macrocycle_rate    - DOMAIN CHECK the generic metrics miss: fraction with a
                         ring of >=12 atoms. This is a macrocycle dataset; a
                         valid, drug-like, novel, non-macrocyclic molecule is
                         still a failure, and nothing else here would say so.
    """
    from rdkit import DataStructs

    from chem_utils import morgan_fps, murcko_scaffold, max_ring_size, brics_fragments

    rng = np.random.default_rng(seed)
    gen_smiles = [s for s in gen_smiles if s]
    if len(gen_smiles) < 5:
        return {}

    sub = gen_smiles if len(gen_smiles) <= sample_size else \
        [gen_smiles[i] for i in rng.choice(len(gen_smiles), sample_size, replace=False)]
    gen_fps = morgan_fps(sub)

    # --- IntDiv1 / IntDiv2 -------------------------------------------------
    sims = []
    for i in range(len(gen_fps) - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(gen_fps[i], gen_fps[i + 1:]))
    sims = np.array(sims) if sims else np.array([0.0])
    intdiv1 = float(1.0 - sims.mean())
    intdiv2 = float(1.0 - np.sqrt((sims ** 2).mean()))

    # --- SNN to training set ----------------------------------------------
    ref = train_smiles if len(train_smiles) <= train_ref_size else \
        [train_smiles[i] for i in rng.choice(len(train_smiles), train_ref_size, replace=False)]
    ref_fps = morgan_fps(ref)
    snn = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)) for fp in gen_fps]) \
        if ref_fps else np.array([])

    # --- scaffolds ---------------------------------------------------------
    scaffs = [s for s in (murcko_scaffold(s) for s in gen_smiles) if s]
    scaf_counts = Counter(scaffs)
    if scaf_counts:
        p = np.array(list(scaf_counts.values()), dtype=float)
        p /= p.sum()
        scaf_H = float(-(p * np.log(p)).sum())
        scaf_H_norm = float(scaf_H / math.log(len(scaf_counts))) if len(scaf_counts) > 1 else 0.0
    else:
        scaf_H = scaf_H_norm = 0.0

    # --- macrocycle rate (domain-specific) ---------------------------------
    ring_sizes = np.array([max_ring_size(s) for s in gen_smiles], dtype=float)
    macro = float((ring_sizes >= 12).mean())

    out = {
        "intdiv1": intdiv1,
        "intdiv2": intdiv2,
        "snn_to_train_mean": float(snn.mean()) if len(snn) else None,
        "snn_to_train_median": float(np.median(snn)) if len(snn) else None,
        "frac_near_duplicate_of_train_tanimoto_0.9": float((snn >= 0.9).mean()) if len(snn) else None,
        "scaffold_uniqueness": float(len(scaf_counts) / len(scaffs)) if scaffs else None,
        "n_unique_scaffolds": int(len(scaf_counts)),
        "scaffold_entropy_nats": scaf_H,
        "scaffold_entropy_normalized": scaf_H_norm,
        "macrocycle_rate": macro,
        "mean_max_ring_size": float(np.nanmean(ring_sizes)) if len(ring_sizes) else None,
        "n_scored": int(len(gen_smiles)),
        "_snn_values": snn.tolist(),  # for plotting; stripped before JSON dump
    }

    # --- BRICS fragment similarity ----------------------------------------
    if compute_frag:
        try:
            g_frags = Counter()
            r_frags = Counter()
            for s in sub:
                g_frags.update(brics_fragments(s))
            for s in ref[:len(sub)]:
                r_frags.update(brics_fragments(s))
            support = sorted(set(g_frags) | set(r_frags))
            if support:
                gv = np.array([g_frags.get(f, 0) for f in support], dtype=float)
                rv = np.array([r_frags.get(f, 0) for f in support], dtype=float)
                denom = np.linalg.norm(gv) * np.linalg.norm(rv)
                out["frag_similarity_cosine"] = float(gv @ rv / denom) if denom > 0 else None
        except Exception as e:
            print(f"[metrics] BRICS fragment similarity skipped ({e})")

    return out


def macrocycle_rate_real(train_smiles, seed=0, n=2000):
    """Reference value: what fraction of the REAL molecules pass the >=12-ring
    check? RDKit's ring perception isn't guaranteed to see every macrocycle, so
    the generated rate should be read against this, not against 100%."""
    from chem_utils import max_ring_size
    rng = np.random.default_rng(seed)
    sub = train_smiles if len(train_smiles) <= n else \
        [train_smiles[i] for i in rng.choice(len(train_smiles), n, replace=False)]
    sizes = np.array([max_ring_size(s) for s in sub], dtype=float)
    return float((sizes >= 12).mean())
