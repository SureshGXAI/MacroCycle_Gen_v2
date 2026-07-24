#!/bin/bash
set -e

CSV=data/All.csv

########################################
# EDA
########################################

python3 eda.py \
    --csv $CSV \
    --out_dir outputs/eda

########################################
# Data Preparation
########################################

declare -A DATASETS=(
    [smiles]=processed
    [selfies]=processed_selfies
)

for rep in "${!DATASETS[@]}"; do
    base=${DATASETS[$rep]}

    echo "========================================"
    echo "Preparing $rep dataset"
    echo "========================================"

    python3 data_prep.py \
        --csv $CSV \
        --out data/${base}.pkl \
        --representation $rep

    python3 data_prep.py \
        --csv $CSV \
        --out data/${base}_300.pkl \
        --representation $rep \
        --max_len 300
done

########################################
# Train
########################################

for rep in "${!DATASETS[@]}"; do

    base=${DATASETS[$rep]}

    if [[ "$rep" == "smiles" ]]; then
        CKPT=checkpoints/model.pt
        RLCKPT=checkpoints/model_rl.pt
    else
        CKPT=checkpoints/model_${rep}.pt
        RLCKPT=checkpoints/model_rl_${rep}.pt
    fi

    echo "========================================"
    echo "Training $rep model"
    echo "========================================"

    python3 train.py \
        --data data/${base}.pkl \
        --epochs 30 \
        --batch_size 128 \
        --ckpt $CKPT

    ####################################
    # Generation (Sampling)
    ####################################

    python3 generate.py \
        --data data/${base}.pkl \
        --ckpt $CKPT \
        --m_type "Macrolide" \
        --MW 550 \
        --LogP 3.0 \
        --n_samples 50 \
        --temperature 0.9 \
        --top_k 30 \
        --top_p 0.95 \
	--out_csv ${base}_generated_molecules.csv
    ####################################
    # Beam Search
    ####################################

    python3 generate.py \
        --data data/${base}.pkl \
        --ckpt $CKPT \
        --m_type "Porphyrin" \
        --decode beam \
        --beam_width 10 \
	--out_csv ${base}_beam_generated_molecules.csv

    ####################################
    # Evaluation
    ####################################

    python3 evaluate.py \
        --data data/${base}.pkl \
        --ckpt $CKPT \
        --n_per_class 200 \
        --out_dir outputs/eval/${rep} \
        --compute_fcd

    ####################################
    # Rejection Sampling (QED)
    ####################################

    python3 optimize.py \
        --mode rejection \
        --data data/${base}.pkl \
        --ckpt $CKPT \
        --m_type "Cyclic Peptide" \
        --objective qed \
        --n_candidates 2000 \
        --top_k_out 20

    ####################################
    # Target Property Optimization
    ####################################

    python3 optimize.py \
        --mode rejection \
        --data data/${base}.pkl \
        --ckpt $CKPT \
        --m_type "Macrolide" \
        --objective target \
        --MW 500 \
        --LogP 2.5 \
        --n_candidates 2000\
	--out_dir outputs/optimize/rejection_${rep}

    ####################################
    # Reinforcement Learning
    ####################################

#    python3 optimize.py \
#        --mode reinforce \
#        --data data/${base}.pkl \
#        --ckpt $CKPT \
#        --m_type "Synthetic Macrocycle" \
#        --objective qed \
#        --rl_steps 500 \
#        --out_ckpt $RLCKPT \
#    	--out_dir outputs/optimize/reinforce_${rep}

done
