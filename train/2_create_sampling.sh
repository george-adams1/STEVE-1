source /opt/conda/bin/activate minedojo
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/georgea/pip_packages:$PYTHONPATH

python steve1/data/sampling/generate_sampling.py \
--type neurips \
--name neurips \
--output_dir data/samplings/ \
--val_frames 10_000 \
--train_frames 30_000