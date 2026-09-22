#!/bin/bash
# Self-contained Habitat rearrange demo job for Turing (headless EGL GPU node).
#   sbatch src/demo/run_turing_gpu.sh
#SBATCH --job-name=hab_demo
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/apatwardhan/habitat_ws/habitat_logs/render_%j.log

set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat
PY=~/miniconda3/envs/habitat/bin/python

cd /scratch/apatwardhan/habitat_ws/WPI_DS551_G10_Final
mkdir -p /scratch/apatwardhan/habitat_ws/habitat_logs videos
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
# glvnd EGL/OpenGL loaders live in the env; make sure they are found on any node
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- ensure render deps, keep numpy<2 for habitat_sim ABI ---
$PY -c "import imageio,cv2,imageio_ffmpeg" 2>/dev/null || \
    $PY -m pip install -q "numpy<2" imageio imageio-ffmpeg opencv-python-headless
$PY -c "import numpy; assert numpy.__version__.startswith('1'), numpy.__version__" || \
    $PY -m pip install -q "numpy<2"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import habitat_sim; print('habitat_sim', habitat_sim.__version__)"

echo "########## INSPECT ##########"
$PY src/demo/render_rearrange_demo.py --inspect --scene apt_0 || true

echo "########## RENDER ##########"
$PY src/demo/render_rearrange_demo.py \
    --scene apt_0 --out videos/rearrange_demo.mp4 \
    --width 1280 --height 720 --fps 30

echo "JOB_DONE"
ls -la videos/
