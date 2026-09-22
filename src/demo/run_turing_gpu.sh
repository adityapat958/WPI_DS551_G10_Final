#!/bin/bash
# Run the Habitat rearrange demo on a Turing GPU node (headless EGL).
# Usage:
#   sbatch src/demo/run_turing_gpu.sh            # full render
#   INSPECT=1 sbatch src/demo/run_turing_gpu.sh  # introspection only
#SBATCH --job-name=hab_demo
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=habitat_logs/render_%j.log

set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /scratch/apatwardhan/habitat_ws/WPI_DS551_G10_Final
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -c "import habitat_sim; print('habitat_sim', habitat_sim.__version__)"

if [ "${INSPECT:-0}" = "1" ]; then
    python src/demo/render_rearrange_demo.py --inspect --scene apt_0
else
    python src/demo/render_rearrange_demo.py \
        --scene apt_0 --out videos/rearrange_demo.mp4 \
        --width 1280 --height 720 --fps 30
fi
echo "JOB_DONE"
