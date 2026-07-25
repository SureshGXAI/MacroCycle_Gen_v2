# Conditional Macrocycle Generator

A GPT-style Transformer that generates novel macrocycle molecules


## Setup

git clone `git@github.com:SureshGXAI/MacroCycle_Gen_v2.git`
cd MacroCycle_Gen_v2
pip install -r requirements.txt

`selfies` and `fcd` in requirements.txt are optional extras.
These are only needed if you use `--representation selfies` or `evaluate.py --compute_fcd`.

## 1. Explore the data first

python3 eda.py --csv data/All.csv --out_dir outputs/eda

Produces 7 plots: class balance (log scale - reveals the 331x imbalance
between Cyclic Peptide and Macrocyclic Polyene), property distributions
overall and by class, MW-vs-LogP scatter, SMILES length distribution (shows
how many molecules the `max_len` cutoff drops), property correlation heatmap,
and clinical-phase breakdown.

## 2. Prepare the data

**Standard: character-level SMILES**

python3 data_prep.py --csv data/All.csv --out data/processed.pkl --representation smiles

**Alternative: SELFIES, guarantees syntactic validity by construction**

python3 data_prep.py --csv data/All.csv --out data/processed_selfies.pkl --representation selfies


What this does: drops junk columns/fixes label typos, validates + canonicalizes
every molecule with RDKit, tokenizes (char-level for SMILES, symbol-level for
SELFIES), builds the 12-dim conditioning vector (6-class one-hot + 6
z-normalized properties), computes **inverse-frequency class weights** for
balanced sampling, and stores the real per-molecule property table for later
generated-vs-real comparison plots.

**SMILES vs SELFIES tradeoff**: SMILES is the conventional/human-readable
representation but the model can emit syntactically broken strings (unmatched
rings/branches). Every valid SELFIES string decodes to a valid molecule by
construction, trading a bit of "natural" chemical-language modeling for a
structural validity guarantee. If SMILES validity plateaus low after training,
switch representations rather than just training longer.

## 3. Train

python3 train.py --data data/processed.pkl --epochs 30 --batch_size 128


## 4. Generate

**Stochastic sampling (diverse)**

python3 generate.py --data data/processed.pkl --ckpt checkpoints/model.pt \
    --m_type "Macrolide" --MW 550 --LogP 3.0 --n_samples 50 \
    --temperature 0.9 --top_k 30 --top_p 0.95

**Deterministic beam search (single target's best-guess candidates)**

python3 generate.py --data data/processed.pkl --ckpt checkpoints/model.pt \
    --m_type "Porphyrin" --decode beam --beam_width 10

Nucleus (top-p) sampling adapts the candidate pool size per step rather than
a fixed top-k, which tends to avoid both repetitive high-confidence loops and
wild derailments; beam search instead gives the model's ranked best guesses
for one target rather than a diverse sample - use it for a final shortlist.

## 5. Full evaluation (metrics + plots)

python3 evaluate.py --data data/processed.pkl --ckpt checkpoints/model.pt \
    --n_per_class 200 --out_dir outputs/eval --compute_fcd

Generates per-class, computes the standard generative-chemistry metric suite,
and plots:
- `property_distributions.png` — generated vs. real MW/LogP/PSA/RotB/HBA/HBD overlaid histograms
- `validity_by_class.png` — validity/uniqueness/novelty bar chart per class (surfaces rare-class weaknesses)
- `qed_sa_distributions.png` — generated-molecule QED and synthetic accessibility (SA) score
- `metrics.json` - all numbers, plus **Fréchet ChemNet Distance** (FCD) if `--compute_fcd` is set

## 6. Optimize - go beyond the training distribution

The base model is a *generator*, not an *optimizer* - on its own it won't
produce molecules better than what it saw in training on some target metric.
`optimize.py` adds two ways to push past that:

**Rejection sampling: generate a large batch, keep the top-K by objective.
Simple, no further training, strong baseline.**

python3 optimize.py --mode rejection --data data/processed.pkl --ckpt checkpoints/model.pt \
    --m_type "Cyclic Peptide" --objective qed --n_candidates 2000 --top_k_out 20

**Target a specific property window instead of maximizing QED**

python3 optimize.py --mode rejection --data data/processed.pkl --ckpt checkpoints/model.pt \
    --m_type "Macrolide" --objective target --MW 500 --LogP 2.5 --n_candidates 2000

**REINFORCE fine-tuning: actually shifts the generator's distribution toward higher-reward molecules, so future sampling is biased without needing large rejection batches every time.**

python3 optimize.py --mode reinforce --data data/processed.pkl --ckpt checkpoints/model.pt \
    --m_type "Synthetic Macrocycle" --objective qed --rl_steps 500 --out_ckpt checkpoints/model_rl.pt
