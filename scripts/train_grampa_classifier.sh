#!/bin/bash
#SBATCH --output=./slurm/slurm_%j.out
#SBATCH --error=./slurm/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a16:1
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
sbatch \
  --export=ALL,DIFFUSION=${DIFFUSION} \
  --job-name=train_grampa_classifier_${DIFFUSION} \
  train_grampa_classifier.sh
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
if [ -z "${DIFFUSION}" ]; then
  echo "DIFFUSION is not set"
  exit 1
fi
T=0
RUN_NAME="grampa_classifier_${DIFFUSION}_T-${T}"

# Train classifier for grampa (peptides) with labels 0-3
# Note: callbacks=classifier_callbacks is commented out
# Note: early_stopping callbacks are commented out
srun python -u -m main \
  mode=train_classifier \
  diffusion=${DIFFUSION} \
  T=${T} \
  data=grampa_classifier \
  data.label_col=label \
  data.num_classes=4 \
  loader.global_batch_size=256 \
  loader.eval_global_batch_size=256 \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  model.length=50 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.val_check_interval=1.0 \
  trainer.max_steps=25_000 \
  wandb.name="grampa-classifier_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/grampa/classifier/${RUN_NAME}"