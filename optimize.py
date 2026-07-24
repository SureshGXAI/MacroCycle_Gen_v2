"""
Property-guided molecule optimization on top of the trained generator.

Two complementary approaches, both included:

1. REJECTION SAMPLING (--mode rejection, default): generate a large batch,
   score every valid molecule against an objective (e.g. maximize QED, or
   minimize distance to a target MW/LogP), and keep the top-K. Simple,
   requires no further training, and is a strong baseline — this is exactly
   how tools like REINVENT's "scaffold-constrained" mode is often used in
   practice for a quick shortlist.

2. REINFORCE FINE-TUNING (--mode reinforce): actually shifts the generator's
   own distribution toward higher-reward molecules via policy-gradient
   fine-tuning (starting from the pretrained checkpoint), so subsequent
   sampling is biased toward the objective without needing large batches of
   rejection sampling every time. Uses a moving-average baseline for variance
   reduction and a small KL-to-pretrained penalty to prevent mode collapse
   into a handful of degenerate high-reward strings (a well-known failure
   mode in RL-for-molecule-generation, e.g. discussed in the REINVENT/ORGAN
   literature).

Examples:
    # Best-of-N shortlist maximizing QED for Cyclic Peptides
    python optimize.py --mode rejection --data data/processed.pkl --ckpt checkpoints/model.pt \
        --m_type "Cyclic Peptide" --objective qed --n_candidates 2000 --top_k_out 20

    # Best-of-N targeting a specific property window
    python optimize.py --mode rejection --data data/processed.pkl --ckpt checkpoints/model.pt \
        --m_type "Macrolide" --objective target --MW 500 --LogP 2.5 --n_candidates 2000

    # Fine-tune the generator itself to prefer higher QED, for a given class
    python optimize.py --mode reinforce --data data/processed.pkl --ckpt checkpoints/model.pt \
        --m_type "Synthetic Macrocycle" --objective qed --rl_steps 500 --out_ckpt checkpoints/model_rl.pt
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import RDLogger

from model import ConditionalSMILESTransformer
from chem_utils import mol_from_sequence, compute_properties
from plotting_utils import plot_optimization_scores

RDLogger.DisableLog("rdApp.*")


def build_condition_vector(data, m_type, prop_overrides):
    class_list = data["class_list"]
    onehot = np.zeros(len(class_list), dtype=np.float32)
    onehot[data["class_to_idx"][m_type]] = 1.0
    prop_vals = []
    for col in data["cond_properties"]:
        stats = data["prop_stats"][col]
        raw = prop_overrides.get(col, stats["mean"])
        prop_vals.append((raw - stats["mean"]) / stats["std"])
    return np.concatenate([onehot, np.array(prop_vals, dtype=np.float32)])


def score_molecule(props, objective, targets=None):
    """Higher is always better, regardless of objective."""
    if objective == "qed":
        return props.get("QED", 0.0)
    elif objective == "sa":  # lower SA is better -> invert
        sa = props.get("SA")
        return -sa if sa is not None else -10.0
    elif objective == "target":
        # negative normalized distance to target property values
        dist = 0.0
        for col, target_val in targets.items():
            if col in props:
                dist += ((props[col] - target_val) / max(1.0, abs(target_val))) ** 2
        return -np.sqrt(dist)
    else:
        raise ValueError(f"Unknown objective: {objective}")


def rejection_sampling(model, data, device, m_type, objective, targets,
                        n_candidates, top_k_out, temperature, top_k, top_p, out_dir):
    cond_vec = build_condition_vector(data, m_type, targets or {})
    cond = torch.tensor(np.tile(cond_vec, (n_candidates, 1)), dtype=torch.float32, device=device)

    representation = data.get("representation", "smiles")
    all_scores, all_rows = [], []
    batch_size = 256
    for start in range(0, n_candidates, batch_size):
        batch_cond = cond[start:start + batch_size]
        seqs = model.generate(batch_cond, data["stoi"], data["itos"], max_new_tokens=data["max_len"],
                               temperature=temperature, top_k=top_k, top_p=top_p, device=device)
        for seq in seqs:
            mol = mol_from_sequence(seq, representation)
            if mol is None:
                continue
            props = compute_properties(mol)
            score = score_molecule(props, objective, targets)
            props.update({"generated_sequence": seq, "score": score})
            all_scores.append(score)
            all_rows.append(props)

    df = pd.DataFrame(all_rows).sort_values("score", ascending=False)
    df_top = df.head(top_k_out)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "rejection_all_candidates.csv"), index=False)
    df_top.to_csv(os.path.join(out_dir, "rejection_top_candidates.csv"), index=False)

    print(f"Generated {n_candidates} candidates, {len(df)} valid ({len(df)/n_candidates:.1%}).")
    print(f"Top-{top_k_out} by '{objective}' objective saved to "
          f"{out_dir}/rejection_top_candidates.csv")
    print(df_top[["generated_sequence", "MW", "LogP", "QED", "SA", "score"]].head(10).to_string(index=False))

    if len(all_scores) >= 20:
        plot_optimization_scores(
            np.array(all_scores), df_top["score"].values, f"{objective} score",
            os.path.join(out_dir, "rejection_score_distribution.png"),
        )
        print(f"Saved score distribution plot -> {out_dir}/rejection_score_distribution.png")


def reinforce_finetune(model, data, device, m_type, objective, targets,
                        rl_steps, rl_batch_size, rl_lr, kl_coef, out_ckpt, out_dir):
    """
    Simple REINFORCE (Williams 1992) fine-tuning: sample sequences from the
    current policy, compute a scalar reward per sequence (from RDKit-computed
    properties), and push up the log-probability of tokens in high-reward
    sequences (relative to a moving-average baseline for variance reduction).

    A KL penalty against the frozen pretrained model's logits discourages the
    policy from collapsing onto a few degenerate high-reward strings — a
    well-documented failure mode when optimizing purely for a proxy reward
    like QED (the model can find "reward hacks": valid-but-degenerate
    molecules that score high on the proxy without being chemically sensible).
    """
    import copy
    representation = data.get("representation", "smiles")
    frozen_model = copy.deepcopy(model)
    for p in frozen_model.parameters():
        p.requires_grad_(False)
    frozen_model.eval()

    cond_vec = build_condition_vector(data, m_type, targets or {})
    cond_template = torch.tensor(cond_vec, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=rl_lr)
    baseline = 0.0
    baseline_momentum = 0.95
    pad_idx = data["stoi"]["<pad>"]
    bos_idx = data["stoi"]["<bos>"]
    eos_idx = data["stoi"]["<eos>"]
    reward_history = []

    print(f"REINFORCE fine-tuning: {rl_steps} steps, batch={rl_batch_size}, "
          f"objective={objective}, class={m_type}")

    for step in range(1, rl_steps + 1):
        model.train()
        cond = cond_template.unsqueeze(0).repeat(rl_batch_size, 1)

        # --- sample a trajectory (with gradient-relevant log-probs tracked) ---
        tokens = torch.full((rl_batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(rl_batch_size, dtype=torch.bool, device=device)
        logp_sum = torch.zeros(rl_batch_size, device=device)
        kl_sum = torch.zeros(rl_batch_size, device=device)

        for _ in range(data["max_len"] - 1):
            T = tokens.size(1)
            padded = torch.full((rl_batch_size, model.max_len), pad_idx, dtype=torch.long, device=device)
            padded[:, :T] = tokens
            logits, _ = model(padded, cond)
            step_logits = logits[:, T - 1, :]
            log_probs = F.log_softmax(step_logits, dim=-1)
            probs = log_probs.exp()

            with torch.no_grad():
                frozen_logits, _ = frozen_model(padded, cond)
                frozen_log_probs = F.log_softmax(frozen_logits[:, T - 1, :], dim=-1)

            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_token = torch.where(finished, torch.full_like(next_token, pad_idx), next_token)

            chosen_logp = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
            step_kl = (probs * (log_probs - frozen_log_probs)).sum(dim=1)

            active = (~finished).float()
            logp_sum = logp_sum + chosen_logp * active
            kl_sum = kl_sum + step_kl * active

            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break

        sequences = model._tokens_to_strings(tokens, data["itos"], eos_idx, pad_idx)

        # --- compute rewards (RDKit; no gradient through this) ---
        rewards = torch.zeros(rl_batch_size, device=device)
        for i, seq in enumerate(sequences):
            mol = mol_from_sequence(seq, representation)
            if mol is None:
                rewards[i] = -1.0  # penalize invalid molecules directly
            else:
                props = compute_properties(mol)
                rewards[i] = score_molecule(props, objective, targets)

        mean_reward = rewards.mean().item()
        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * mean_reward
        advantage = rewards - baseline

        loss = -(advantage.detach() * logp_sum).mean() + kl_coef * kl_sum.mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        reward_history.append(mean_reward)
        if step % 10 == 0 or step == 1:
            valid_frac = sum(1 for s in sequences if mol_from_sequence(s, representation) is not None) / rl_batch_size
            print(f"  step {step:4d}/{rl_steps}  mean_reward={mean_reward:.3f}  "
                  f"baseline={baseline:.3f}  valid_frac={valid_frac:.2%}  loss={loss.item():.3f}")

    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": None}, out_ckpt)
    print(f"Saved RL-fine-tuned checkpoint -> {out_ckpt}  "
          f"(NOTE: reload config from original pretrained ckpt when loading this one)")

    os.makedirs(out_dir, exist_ok=True)
    if len(reward_history) > 10:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(reward_history)
        ax.set_xlabel("RL step")
        ax.set_ylabel(f"mean batch reward ({objective})")
        ax.set_title("REINFORCE fine-tuning: reward over time")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "reinforce_reward_curve.png"), dpi=150)
        plt.close(fig)
        print(f"Saved reward curve -> {out_dir}/reinforce_reward_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rejection", "reinforce"], default="rejection")
    ap.add_argument("--data", default="data/processed.pkl")
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--m_type", required=True)
    ap.add_argument("--objective", choices=["qed", "sa", "target"], default="qed")
    ap.add_argument("--MW", type=float, default=None)
    ap.add_argument("--LogP", type=float, default=None)
    ap.add_argument("--HBA", type=float, default=None)
    ap.add_argument("--HBD", type=float, default=None)
    ap.add_argument("--PSA", type=float, default=None)
    ap.add_argument("--RotB", type=float, default=None)
    # rejection sampling args
    ap.add_argument("--n_candidates", type=int, default=1000)
    ap.add_argument("--top_k_out", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--top_p", type=float, default=0.95)
    # reinforce args
    ap.add_argument("--rl_steps", type=int, default=300)
    ap.add_argument("--rl_batch_size", type=int, default=32)
    ap.add_argument("--rl_lr", type=float, default=1e-5)
    ap.add_argument("--kl_coef", type=float, default=0.02)
    ap.add_argument("--out_ckpt", default="checkpoints/model_rl.pt")
    ap.add_argument("--out_dir", default="outputs/optimize")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = ConditionalSMILESTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    targets = {}
    for col in data["cond_properties"]:
        val = getattr(args, col)
        if val is not None:
            targets[col] = val
    if args.objective == "target" and not targets:
        raise ValueError("--objective target requires at least one property flag (e.g. --MW 500)")

    if args.mode == "rejection":
        model.eval()
        rejection_sampling(model, data, device, args.m_type, args.objective, targets,
                            args.n_candidates, args.top_k_out, args.temperature,
                            args.top_k, args.top_p, args.out_dir)
    else:
        reinforce_finetune(model, data, device, args.m_type, args.objective, targets,
                            args.rl_steps, args.rl_batch_size, args.rl_lr, args.kl_coef,
                            args.out_ckpt, args.out_dir)


if __name__ == "__main__":
    main()
