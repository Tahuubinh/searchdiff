#!/bin/bash
#SBATCH --output=./slurm/slurm_%j.out
#SBATCH --error=./slurm/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a4500:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=billyta2005@gmail.com

<<comment
#  Usage:
cd scripts/
DIFFUSION=<absorbing_state|uniform>
PROP=<qed|ring_count>
sbatch \
  --export=ALL,DIFFUSION=${DIFFUSION},PROP=${PROP} \
  --job-name=train_qm9_classifier_${PROP}_${DIFFUSION} \
  train_qm9_classifier.sh
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
#  - DIFFUSION (absorbing_state or uniform)
#  - PROP (qed or ring_count)
if [ -z "${DIFFUSION}" ]; then
  echo "DIFFUSION is not set"
  exit 1
fi
if [ -z "${PROP}" ]; then
  echo "PROP is not set"
  exit 1
fi
T=0
RUN_NAME="${PROP}_${DIFFUSION}_T2-${T}"

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  mode=train_classifier \
  diffusion=${DIFFUSION} \
  T=${T} \
  data=qm9 \
  data.cache_dir="${PWD}/data/cache/qm9" \
  data.label_col="${PROP}" \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=2048 \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.val_check_interval=1.0 \
  trainer.max_steps=25_000 \
  wandb.group=train_classifier \
  wandb.name="qm9-classifier_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/qm9/classifier/${RUN_NAME}"