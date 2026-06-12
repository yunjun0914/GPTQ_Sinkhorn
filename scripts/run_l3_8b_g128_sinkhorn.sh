#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=gptqs_l3_8b_g128_sinkhorn
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --output=/home/yunjun0914/GPTQ_Sinkhorn/logs/%x_%j.out
#SBATCH --error=/home/yunjun0914/GPTQ_Sinkhorn/logs/%x_%j.err

MODEL=/home/yunjun0914/models/llama3-8b
QUANT_DIR=/home/yunjun0914/GPTQ_Sinkhorn/outputs/l3_8b_g128_sinkhorn
RECON_DIR=/home/yunjun0914/GPTQ_Sinkhorn/outputs/l3_8b_g128_sinkhorn_recon

export CUDA_HOME=/usr/local/cuda
export PATH=/home/yunjun0914/bin:/home/yunjun0914/yunjun_env/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64

mkdir -p $QUANT_DIR
mkdir -p ~/GPTQ_Sinkhorn/logs
cd ~/GPTQ_Sinkhorn

python quantize.py $MODEL \
    --output_dir $QUANT_DIR \
    --bits 4 \
    --group_size 128 \
    --sinkhorn \
    --n_samples 128 \
    --seqlen 2048 \
    --device cuda

python reconstruct.py \
    --quantized_dir $QUANT_DIR \
    --output_dir $RECON_DIR

PPL_OUT=$(python eval_ppl.py --model_dir $RECON_DIR --dataset wikitext2 c4 2>/dev/null)
W2=$(echo "$PPL_OUT" | grep "^wikitext2" | awk '{print $NF}')
C4=$(echo "$PPL_OUT" | grep "^c4" | awk '{print $NF}')

echo "wikitext2: $W2 | c4: $C4"
send_ch yunjun0914_exp "[DONE] judy $SLURM_JOB_NAME ($SLURM_JOB_ID)
LLaMA-3-8B 4bit g128 sinkhorn only
wt2=$W2 c4=$C4"
