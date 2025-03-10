#!/bin/bash
#SBATCH --job-name=steve_1_test
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --account=def-irina
#SBATCH --mail-user=george.adamopoulos13@gmail.com
#SBATCH --mail-type=ALL

module load apptainer

cd /scratch/georgea/STEVE-1

# Use exec instead of shell for batch scripts
apptainer exec --nv --cleanenv steve2.sif bash -c '
source /opt/conda/bin/activate minedojo
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/georgea/STEVE-1:/scratch/georgea/pip_files_3:$PYTHONPATH
export WANDB_MODE=disabled

accelerate launch --num_processes 1 --mixed_precision fp16 steve1/training/train.py \
--in_model data/weights/vpt/2x.model \
--in_weights data/weights/vpt/rl-from-foundation-2x.weights \
--out_weights data/weights/steve1/trained_with_script.weights \
--trunc_t 20 \
--T 160 \
--batch_size 8 \
--gradient_accumulation_steps 4 \
--num_workers 1 \
--weight_decay 0.039428 \
--n_frames 100_000_000 \
--learning_rate 4e-5 \
--warmup_frames 10_000_000 \
--p_uncond 0.1 \
--min_btwn_goals 15 \
--max_btwn_goals 200 \
--checkpoint_dir data/training_checkpoint \
--val_freq 2000 \
--val_freq_begin 200 \
--val_freq_switch_steps 500 \
--val_every_nth 10 \
--save_each_val False \
--sampling neurips \
--sampling_dir data/samplings/ \
--snapshot_every_n_frames 50_000_000
'