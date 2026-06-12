#!/bin/bash
#SBATCH --partition=rabbit
#SBATCH --job-name=gptqs_l3_8b_viz_weight
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=/home/yunjun0914/GPTQ_Sinkhorn/logs/%x_%j.out
#SBATCH --error=/home/yunjun0914/GPTQ_Sinkhorn/logs/%x_%j.err

MODEL=/home/yunjun0914/models/llama3-8b
OUT=/home/yunjun0914/GPTQ_Sinkhorn/logs/l3_8b_weight_transform_viz_${SLURM_JOB_ID}.png
OUT_ABS=/home/yunjun0914/GPTQ_Sinkhorn/logs/l3_8b_weight_transform_viz_abs_${SLURM_JOB_ID}.png

export PATH=/home/yunjun0914/bin:/home/yunjun0914/yunjun_env/bin:/usr/local/bin:/usr/bin:/bin

mkdir -p ~/GPTQ_Sinkhorn/logs
cd ~/GPTQ_Sinkhorn

python visualize_weight_transforms.py $MODEL \
    --stride 1 \
    --elev 30 --azim -60 \
    --out $OUT

python visualize_weight_transforms.py $MODEL \
    --stride 1 \
    --abs \
    --elev 30 --azim -60 \
    --out $OUT_ABS

echo "Saved: $OUT  and  $OUT_ABS"
send_ch yunjun0914_exp "[DONE] LLaMA-3-8B weight viz ($SLURM_JOB_ID) → l3_8b_weight_transform_viz_${SLURM_JOB_ID}.png + abs"
