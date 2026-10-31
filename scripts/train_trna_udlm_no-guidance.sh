#!/bin/bash
#SBATCH --output=./slurm/slurm_%j.out
#SBATCH --error=./slurm/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=billyta2005@gmail.com

<<comment
#  Usage:
cd scripts/
MODEL=<ar|mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=train_trna_no-guidance_${MODEL} \
  train_trna_udlm_no-guidance.sh
comment

# Setup environment
set -e
source /u/nzj6jt/miniconda3/etc/profile.d/conda.sh
conda activate sedd
cd ../ || exit  
export WANDB_MODE=disabled
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (ar, mdlm, udlm)
#  - USE_SIMPLE_CE_LOSS (True, False; optional, default: False)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
if [ -z "${USE_SIMPLE_CE_LOSS}" ]; then
  USE_SIMPLE_CE_LOSS=False
fi
RUN_NAME="${MODEL}_no-guidance"
if [ "${USE_SIMPLE_CE_LOSS}" = "True" ]; then
  RUN_NAME="${RUN_NAME}_simple-ce"
fi

if [ "${MODEL}" = "ar" ]; then
  # AR
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="ar"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
elif [ "${MODEL}" = "mdlm" ]; then
  # MDLM
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="subs"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
elif [ "${MODEL}" = "udlm" ]; then
  # UDLM
  DIFFUSION="uniform"
  PARAMETERIZATION="d3pm"
  T=0
  TIME_COND=True
  ZERO_RECON_LOSS=True
else
  echo "MODEL must be one of ar, mdlm, udlm"
  exit 1
fi

# To enable preemption re-loading, set `hydra.run.dir` or
# `checkpointing.save_dir` explicitly.
srun python -u -m main \
  diffusion="${DIFFUSION}" \
  parameterization="${PARAMETERIZATION}" \
  T=${T} \
  time_conditioning=${TIME_COND} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=trna \
  eval.generate_samples=False \
  loader.global_batch_size=256 \
  loader.eval_global_batch_size=256 \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  backbone="dit" \
  model=small \
  model.length=110 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  training.guidance=null \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  training.compute_loss_on_pad_tokens=True \
  training.use_simple_ce_loss=${USE_SIMPLE_CE_LOSS} \
  trainer.max_steps=25_000 \
  trainer.val_check_interval=1.0 \
  wandb.name="trna_udlm_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/trna/${RUN_NAME}"
