"""
Train the conditional SMILES Transformer.

Run (Colab/GPU recommended):
    python train.py --data data/processed.pkl --epochs 30 --batch_size 128
"""
import argparse
import pickle
import time

import numpy as np
import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model import ConditionalSMILESTransformer
from chem_utils import validity_uniqueness_novelty

RDLogger.DisableLog("rdApp.*")


class SmilesDataset(Dataset):
    def __init__(self, token_ids, cond, indices):
        self.token_ids = token_ids[indices]
        self.cond = cond[indices]

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, i):
        ids = self.token_ids[i]
        x = ids[:-1]   # input:  BOS c1 c2 ... (all but last)
        y = ids[1:]    # target: c1 c2 ... EOS (shifted by one)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), torch.from_numpy(self.cond[i].copy())


def sample_and_report(model, data, device, n=16):
    """Sample n random real conditioning vectors from val set and generate."""
    idx = np.random.choice(data["val_idx"], size=min(n, len(data["val_idx"])), replace=False)
    cond = torch.tensor(data["cond"][idx], dtype=torch.float32, device=device)
    gen = model.generate(
        cond, data["stoi"], data["itos"],
        max_new_tokens=data["max_len"], temperature=1.0, top_k=30, device=device,
    )
    rep = data.get("representation", "smiles")
    train_set = set(data["smiles"])
    valid_smiles, vr, uniq, novel = validity_uniqueness_novelty(gen, rep, train_set)
    print(f"  [sample] validity={vr:.2%}  uniqueness={uniq:.2%}  novelty={novel:.2%}")
    for s in gen[:5]:
        print(f"    {s}")
    return vr


def per_class_validity_report(model, data, device, n_per_class=20):
    """
    Generate n_per_class molecules for EVERY class using that class's mean
    properties, and report validity per class. This is the key diagnostic for
    imbalance: overall validity can look fine while rare classes (e.g.
    Macrocyclic Polyene, 0.16% of data) generate poorly.
    """
    print("  [per-class validity]")
    n_classes = len(data["class_list"])
    val_cond = data["cond"][data["val_idx"]]
    val_class_idx = np.argmax(val_cond[:, :n_classes], axis=1)  # class label per val row

    for cls in data["class_list"]:
        cls_idx = data["class_to_idx"][cls]
        matches = data["val_idx"][val_class_idx == cls_idx]
        # fall back to a synthetic mean-property vector for this class if none in val split
        if len(matches) == 0:
            cond_vec = np.zeros(data["cond"].shape[1], dtype=np.float32)
            cond_vec[cls_idx] = 1.0
            cond = torch.tensor(np.tile(cond_vec, (n_per_class, 1)), dtype=torch.float32, device=device)
        else:
            chosen = np.random.choice(matches, size=min(n_per_class, len(matches)),
                                       replace=len(matches) < n_per_class)
            cond = torch.tensor(data["cond"][chosen], dtype=torch.float32, device=device)
        gen = model.generate(cond, data["stoi"], data["itos"], max_new_tokens=data["max_len"],
                              temperature=1.0, top_k=30, device=device)
        rep = data.get("representation", "smiles")
        _, vr, _, _ = validity_uniqueness_novelty(gen, rep, set(data["smiles"]))
        print(f"    {cls:22s} validity={vr:.2%}  (n={len(gen)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed.pkl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--sample_every", type=int, default=2, help="epochs between sampling reports")
    ap.add_argument("--per_class_report_every", type=int, default=10,
                     help="epochs between full per-class validity breakdowns (slower, n_classes x n_per_class samples)")
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--balanced_sampling", action="store_true", default=True,
                     help="Use inverse-class-frequency WeightedRandomSampler so rare classes "
                          "(e.g. Macrocyclic Polyene, 0.16%% of data) are seen as often as common ones "
                          "each epoch, instead of being drowned out by Cyclic Peptide (54%%).")
    ap.add_argument("--no_balanced_sampling", dest="balanced_sampling", action="store_false")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    train_ds = SmilesDataset(data["token_ids"], data["cond"], data["train_idx"])
    val_ds = SmilesDataset(data["token_ids"], data["cond"], data["val_idx"])

    if args.balanced_sampling and "per_sample_weight" in data:
        train_weights = data["per_sample_weight"][data["train_idx"]]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(train_weights, dtype=torch.double),
            num_samples=len(train_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True)
        print("Balanced sampling ON: rare classes (e.g. Macrocyclic Polyene) "
              "upweighted ~100x relative to Cyclic Peptide during training.")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print("Balanced sampling OFF: natural class frequency used (Cyclic Peptide dominates batches).")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    vocab_size = len(data["stoi"])
    cond_dim = data["cond"].shape[1]
    seq_len = data["max_len"] - 1  # model consumes/produces max_len-1 token pairs (x,y)

    model = ConditionalSMILESTransformer(
        vocab_size=vocab_size,
        cond_dim=cond_dim,
        max_len=seq_len,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=data["stoi"]["<pad>"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for x, y, cond in train_loader:
            x, y, cond = x.to(device), y.to(device), cond.to(device)
            _, loss = model(x, cond, targets=y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        model.eval()
        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for x, y, cond in val_loader:
                x, y, cond = x.to(device), y.to(device), cond.to(device)
                _, loss = model(x, cond, targets=y)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(1, n_val_batches)

        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({dt:.1f}s)")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample_and_report(model, data, device)

        if epoch % args.per_class_report_every == 0 or epoch == args.epochs:
            per_class_validity_report(model, data, device)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": dict(
                        vocab_size=vocab_size, cond_dim=cond_dim, max_len=seq_len,
                        d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
                        d_ff=args.d_ff, dropout=args.dropout, pad_idx=data["stoi"]["<pad>"],
                    ),
                },
                args.ckpt,
            )
            print(f"  saved new best checkpoint (val_loss={val_loss:.4f}) -> {args.ckpt}")

    print("Training complete.")


if __name__ == "__main__":
    main()
