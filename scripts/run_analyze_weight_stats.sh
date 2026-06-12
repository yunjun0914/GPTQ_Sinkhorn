#!/bin/bash
#SBATCH --partition=rabbit
#SBATCH --job-name=gptqs_analyze_stats
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yunjun0914/GPTQ_Sinkhorn/logs/%x_%j.out
#SBATCH --error=/home/yunjun0914/GPTQ_Sinkhorn/logs/%x_%j.err

MODEL=/home/yunjun0914/models/llama2-7b

export PATH=/home/yunjun0914/bin:/home/yunjun0914/yunjun_env/bin:/usr/local/bin:/usr/bin:/bin

mkdir -p ~/GPTQ_Sinkhorn/logs
cd ~/GPTQ_Sinkhorn

python analyze_weight_stats.py $MODEL \
    > ~/GPTQ_Sinkhorn/logs/weight_stats_${SLURM_JOB_ID}.txt 2>/dev/null

echo "Done. Results at ~/GPTQ_Sinkhorn/logs/weight_stats_${SLURM_JOB_ID}.txt"
send_ch yunjun0914_exp "[DONE] weight stats analysis ($SLURM_JOB_ID) — check logs/weight_stats_${SLURM_JOB_ID}.txt"
