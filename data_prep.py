"""
Data preparation for the conditional macrocycle generator.

Reads All.csv, cleans it, validates + canonicalizes every molecule with
RDKit, tokenizes it (character-level SMILES OR token-level SELFIES),
builds a per-molecule conditioning vector (class one-hot + normalized
MW/LogP/HBA/HBD/PSA/RotB), and computes class weights for imbalance handling.

Representation choice (--representation smiles|selfies):
  - smiles (default): standard, human-readable, but the model can generate
    syntactically invalid strings (unmatched rings/branches, bad valence).
  - selfies: every SELFIES string decodes to a *chemically valid* molecule by
    construction (no unmatched parens/rings are possible in the grammar), so
    generation-time validity is structurally much higher. Trade-off: slightly
    less direct correspondence to conventional chemical notation, and
    requires `pip install selfies`.

Run:
    python data_prep.py --csv data/All.csv --out data/processed.pkl --representation smiles
    python data_prep.py --csv data/All.csv --out data/processed_selfies.pkl --representation selfies
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit parsing warnings

# Special tokens
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

# Property columns used as continuous conditioning signals.
COND_PROPERTIES = ["MW", "LogP", "HBA", "HBD", "PSA", "RotB"]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup: drop junk column, fix label typos, drop missing SMILES."""
    df = df.copy()
    if "Unnamed: 5" in df.columns:
        df = df.drop(columns=["Unnamed: 5"])

    df["M_type"] = df["M_type"].astype(str).str.strip()
    df["M_type"] = df["M_type"].replace({"Cyclic peptide": "Cyclic Peptide"})

    df = df.dropna(subset=["SMILES"]).reset_index(drop=True)
    return df


def validate_and_canonicalize(smiles_list):
    """Parse each SMILES with RDKit; return canonical SMILES or None if invalid."""
    canon = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            canon.append(None)
        else:
            try:
                canon.append(Chem.MolToSmiles(mol, canonical=True))
            except Exception:
                canon.append(None)
    return canon


def smiles_to_selfies(smiles_list):
    """Convert canonical SMILES -> SELFIES strings. None on failure."""
    import selfies as sf
    out = []
    for smi in smiles_list:
        try:
            out.append(sf.encoder(smi))
        except Exception:
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# Tokenization: SMILES = character-level. SELFIES = symbol-level (each
# [Bracket] group is one token, per the SELFIES grammar).
# ---------------------------------------------------------------------------

def tokenize_smiles(s):
    return list(s)


def tokenize_selfies(s):
    import selfies as sf
    return list(sf.split_selfies(s))


def build_vocab(sequences_of_tokens):
    vocab_tokens = set()
    for toks in sequences_of_tokens:
        vocab_tokens.update(toks)
    vocab = SPECIAL_TOKENS + sorted(vocab_tokens)
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def encode_tokens(tokens, stoi, max_len):
    """BOS + tokens + EOS, padded/truncated to max_len (ids as list[int])."""
    ids = [stoi[BOS]] + [stoi.get(t, stoi[UNK]) for t in tokens] + [stoi[EOS]]
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [stoi[EOS]]
    pad_len = max_len - len(ids)
    ids = ids + [stoi[PAD]] * pad_len
    return ids


def build_conditioning(df, class_list):
    """
    Conditioning vector per molecule = [one-hot class] + [normalized properties].
    Properties are z-score normalized; stats are saved so generation time can
    convert a human target (e.g. MW=500) into the same normalized space.
    """
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    n = len(df)
    n_classes = len(class_list)

    onehot = np.zeros((n, n_classes), dtype=np.float32)
    for i, c in enumerate(df["M_type"]):
        onehot[i, class_to_idx[c]] = 1.0

    prop_stats = {}
    prop_matrix = np.zeros((n, len(COND_PROPERTIES)), dtype=np.float32)
    for j, col in enumerate(COND_PROPERTIES):
        vals = df[col].astype(float).values
        mean, std = np.nanmean(vals), np.nanstd(vals) + 1e-6
        vals = np.nan_to_num(vals, nan=mean)
        prop_matrix[:, j] = (vals - mean) / std
        prop_stats[col] = {"mean": float(mean), "std": float(std)}

    cond = np.concatenate([onehot, prop_matrix], axis=1)
    return cond, class_to_idx, prop_stats


def compute_class_weights(df, class_list, class_to_idx):
    """
    Inverse-frequency class weights, for use as a WeightedRandomSampler in
    training (addresses the ~330x imbalance between Cyclic Peptide (54%) and
    Macrocyclic Polyene (0.16%)). Weight_c = N / (n_classes * count_c), the
    standard 'balanced' formula (sklearn's compute_class_weight uses the same).
    """
    counts = df["M_type"].value_counts()
    n = len(df)
    n_classes = len(class_list)
    weight_by_class = {c: n / (n_classes * counts[c]) for c in class_list}
    per_sample_weight = df["M_type"].map(weight_by_class).values.astype(np.float32)
    return weight_by_class, per_sample_weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/All.csv")
    ap.add_argument("--out", default="data/processed.pkl")
    ap.add_argument("--representation", choices=["smiles", "selfies"], default="smiles")
    ap.add_argument("--max_len", type=int, default=200,
                     help="Max sequence length in tokens (BOS+tokens+EOS). "
                          "Molecules longer than this are dropped, not truncated mid-string, "
                          "to avoid teaching the model to emit broken molecules.")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    df = clean_dataframe(df)
    print(f"  {len(df)} rows after basic cleanup")

    print("Validating & canonicalizing SMILES with RDKit ...")
    canon = validate_and_canonicalize(df["SMILES"].tolist())
    df["canonical_smiles"] = canon
    n_before = len(df)
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    print(f"  {len(df)}/{n_before} valid molecules "
          f"({n_before - len(df)} dropped as unparseable)")

    if args.representation == "selfies":
        print("Converting canonical SMILES -> SELFIES ...")
        selfies_list = smiles_to_selfies(df["canonical_smiles"].tolist())
        df["selfies"] = selfies_list
        n_before2 = len(df)
        df = df.dropna(subset=["selfies"]).reset_index(drop=True)
        print(f"  {len(df)}/{n_before2} molecules successfully SELFIES-encoded "
              f"({n_before2 - len(df)} dropped)")
        seq_col = "selfies"
        tokenize_fn = tokenize_selfies
    else:
        seq_col = "canonical_smiles"
        tokenize_fn = tokenize_smiles

    # Drop rows missing conditioning properties needed downstream
    df = df.dropna(subset=COND_PROPERTIES).reset_index(drop=True)

    print(f"Tokenizing ({args.representation}) ...")
    token_seqs = [tokenize_fn(s) for s in df[seq_col]]
    lengths = np.array([len(t) + 2 for t in token_seqs])  # +BOS +EOS
    keep = lengths <= args.max_len
    print(f"  Dropping {(~keep).sum()} molecules longer than max_len={args.max_len}")
    df = df[keep].reset_index(drop=True)
    token_seqs = [t for t, k in zip(token_seqs, keep) if k]

    class_list = sorted(df["M_type"].unique().tolist())
    print(f"Classes ({len(class_list)}): {class_list}")

    stoi, itos = build_vocab(token_seqs)
    print(f"Vocab size: {len(stoi)} ({args.representation})")

    token_ids = np.array(
        [encode_tokens(t, stoi, args.max_len) for t in token_seqs],
        dtype=np.int64,
    )

    cond, class_to_idx, prop_stats = build_conditioning(df, class_list)
    weight_by_class, per_sample_weight = compute_class_weights(df, class_list, class_to_idx)
    print("Class weights (inverse-frequency, for balanced sampling):")
    for c in class_list:
        print(f"    {c:22s}  n={int((df['M_type']==c).sum()):6d}  weight={weight_by_class[c]:.3f}")

    # Train / val split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    n_val = max(1, int(len(df) * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    processed = {
        "representation": args.representation,
        "token_ids": token_ids,
        "cond": cond,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "stoi": stoi,
        "itos": itos,
        "class_list": class_list,
        "class_to_idx": class_to_idx,
        "prop_stats": prop_stats,
        "cond_properties": COND_PROPERTIES,
        "max_len": args.max_len,
        "smiles": df["canonical_smiles"].tolist(),  # always kept for novelty/eval checks
        "sequences": df[seq_col].tolist(),           # native representation strings
        "per_sample_weight": per_sample_weight,
        "class_weight": weight_by_class,
        # full property table, kept for later distribution-comparison plots
        "real_properties": df[COND_PROPERTIES + ["M_type"]].reset_index(drop=True),
    }

    with open(args.out, "wb") as f:
        pickle.dump(processed, f)

    print(f"Saved processed dataset to {args.out}")
    print(f"  train={len(train_idx)}  val={len(val_idx)}")
    print(f"  token dim={token_ids.shape[1]}  cond dim={cond.shape[1]}")


if __name__ == "__main__":
    main()
