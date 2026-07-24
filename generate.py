"""
Generate new macrocycle molecules from a trained checkpoint, steered by
molecule class and target property values. Supports both stochastic
sampling (temperature/top-k/top-p, diverse) and beam search (deterministic
best-guess) decoding, and both SMILES and SELFIES model representations.

Examples:
    # Stochastic sampling (nucleus + top-k combined)
    python generate.py --data data/processed.pkl --ckpt checkpoints/model.pt \
        --m_type "Macrolide" --MW 550 --LogP 3.0 --HBA 8 --HBD 2 --PSA 120 --RotB 6 \
        --n_samples 50 --temperature 0.9 --top_k 30 --top_p 0.95

    # Deterministic beam search (single target, best-guess candidates)
    python generate.py --data data/processed.pkl --ckpt checkpoints/model.pt \
        --m_type "Porphyrin" --decode beam --beam_width 10
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from model import ConditionalSMILESTransformer
from chem_utils import decode_sequence, mol_from_sequence, compute_properties

RDLogger.DisableLog("rdApp.*")


def build_condition_vector(data, m_type, prop_overrides):
    class_list = data["class_list"]
    if m_type not in class_list:
        raise ValueError(f"Unknown m_type '{m_type}'. Choices: {class_list}")

    onehot = np.zeros(len(class_list), dtype=np.float32)
    onehot[data["class_to_idx"][m_type]] = 1.0

    prop_vals = []
    for col in data["cond_properties"]:
        stats = data["prop_stats"][col]
        raw = prop_overrides.get(col, stats["mean"])  # default to dataset mean if unset
        z = (raw - stats["mean"]) / stats["std"]
        prop_vals.append(z)

    cond = np.concatenate([onehot, np.array(prop_vals, dtype=np.float32)])
    return cond


def rows_from_sequences(sequences, representation, train_smiles_set, scores=None):
    rows = []
    for i, seq in enumerate(sequences):
        smi = decode_sequence(seq, representation)
        mol = mol_from_sequence(seq, representation)
        valid = mol is not None
        row = {
            "generated_sequence": seq,
            "decoded_smiles": smi,
            "valid": valid,
            "novel": bool(valid and smi not in train_smiles_set),
        }
        if scores is not None:
            row["beam_logprob"] = scores[i]
        if valid:
            row.update(compute_properties(mol))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed.pkl")
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--m_type", required=True,
                     help="e.g. 'Cyclic Peptide', 'Macrolide', 'Synthetic Macrocycle', "
                          "'Porphyrin', 'Macrocyclic Polyene', 'Other'")
    ap.add_argument("--MW", type=float, default=None)
    ap.add_argument("--LogP", type=float, default=None)
    ap.add_argument("--HBA", type=float, default=None)
    ap.add_argument("--HBD", type=float, default=None)
    ap.add_argument("--PSA", type=float, default=None)
    ap.add_argument("--RotB", type=float, default=None)
    ap.add_argument("--decode", choices=["sample", "beam"], default="sample")
    ap.add_argument("--n_samples", type=int, default=20, help="used when --decode sample")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--top_p", type=float, default=0.95,
                     help="nucleus sampling threshold; set to 1.0 or omit to disable")
    ap.add_argument("--beam_width", type=int, default=10, help="used when --decode beam")
    ap.add_argument("--out_csv", default="outputs/generated_molecules.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    representation = data.get("representation", "smiles")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = ConditionalSMILESTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prop_overrides = {}
    for col in data["cond_properties"]:
        val = getattr(args, col)
        if val is not None:
            prop_overrides[col] = val

    cond_vec = build_condition_vector(data, args.m_type, prop_overrides)
    train_smiles = set(data["smiles"])

    if args.decode == "beam":
        cond_single = torch.tensor(cond_vec, dtype=torch.float32)
        beam_results = model.beam_search(
            cond_single, data["stoi"], data["itos"],
            max_new_tokens=data["max_len"], beam_width=args.beam_width, device=device,
        )
        sequences = [s for s, _ in beam_results]
        scores = [sc for _, sc in beam_results]
        rows = rows_from_sequences(sequences, representation, train_smiles, scores=scores)
    else:
        cond = torch.tensor(
            np.tile(cond_vec, (args.n_samples, 1)), dtype=torch.float32, device=device
        )
        top_p = args.top_p if args.top_p < 1.0 else None
        sequences = model.generate(
            cond, data["stoi"], data["itos"],
            max_new_tokens=data["max_len"], temperature=args.temperature,
            top_k=args.top_k, top_p=top_p, device=device,
        )
        rows = rows_from_sequences(sequences, representation, train_smiles)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    n_valid = df["valid"].sum()
    n_novel = df["novel"].sum()
    n_unique = df.loc[df["valid"], "decoded_smiles"].nunique() if n_valid else 0
    print(f"Requested class: {args.m_type}   overrides: {prop_overrides}   "
          f"representation: {representation}   decode: {args.decode}")
    print(f"Generated {len(df)} samples -> valid={n_valid} ({n_valid/len(df):.1%})  "
          f"unique={n_unique}  novel={n_novel}")
    print(f"Saved to {args.out_csv}")
    if n_valid:
        cols = ["decoded_smiles", "MW", "LogP", "QED", "SA"]
        if "beam_logprob" in df.columns:
            cols.append("beam_logprob")
        print(df.loc[df["valid"], cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
