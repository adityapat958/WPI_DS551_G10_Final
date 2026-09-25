# vizjob hook — DS551 Habitat portfolio renders on WPI Turing.
# Sourced locally (VJ_* + vj_sync) and on the host (job_* functions).
VJ_PROJECT=ds551hab
VJ_HOST=turing.wpi.edu
VJ_DIR=/scratch/apatwardhan/habitat_ws/WPI_DS551_G10_Final

# push code only (data/ + models/ already live on the host)
vj_sync() { rsync -a -e ssh ./src/demo/ "$VJ_HOST:$VJ_DIR/src/demo/"; }

# RTX PRO 6000 Blackwell nodes hang habitat-sim's EGL init -> never schedule there
_VJ_EXCL=gpu-6-01,gpu-6-02,gpu-6-03,gpu-6-04,gpu-6-05,gpu-6-06,gpu-6-07,gpu-6-08,gpu-6-09,gpu-6-10,gpu-6-11,gpu-6-12,gpu-6-13,gpu-6-14,gpu-6-15,gpu-6-16,gpu-6-17,gpu-6-18,gpu-6-19,gpu-6-20

_vj_gpu() {  # _vj_gpu MINUTES CMD...
  local mins="$1"; shift
  export PATH=/cm/shared/apps/slurm/current/bin:$PATH
  srun --partition=short,quick,long --gres=gpu:1 --exclude="$_VJ_EXCL" --cpus-per-task=4 --mem=24G \
       --time="00:${mins}:00" --job-name=hab_v3 --export=ALL \
       bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate /scratch/apatwardhan/envs/habitat && \
                 export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet && \
                 nvidia-smi --query-gpu=name --format=csv,noheader && cd $VJ_DIR && $*"
}

job_selftest() { _vj_gpu 15 python src/demo/render_v3.py --selftest --out videos/v3/selftest.mp4 "$@"; }
job_render()   { _vj_gpu 40 python src/demo/render_v3.py --out videos/v3/fetch_bedroom_to_kitchen.mp4 "$@"; }
job_final()    { _vj_gpu 59 python src/demo/render_v3.py --width 1920 --height 1080 --out videos/v3/fetch_bedroom_to_kitchen_1080p.mp4 "$@"; }

# pretrained 2022 challenge skills (nav->pick->nav->place) on rearrange_easy val episodes
job_rl() {
  _vj_gpu 59 python src/rl_skills/rollout_skills.py --models-dir data/models --split val \
      --num-episodes ${RL_EPS:-12} --out-dir videos/rl_skills "$@" \
    && python - <<'PY'
import glob, os, sys
sys.path.insert(0, os.environ.get("VIZJOB_LIB", os.path.expanduser("~/.vizjob/lib")))
import vizjob_hook as vj
v = sorted(glob.glob("videos/rl_skills/*.mp4"))
succ = [p for p in v if "SUCC" in p]
vj.post_text(f"RL skills: {len(succ)}/{len(v)} episodes succeeded")
for p in (succ or v)[:3]:
    vj.post_image(p, "2022 pretrained skills | " + os.path.basename(p))
PY
}
