#!/usr/bin/env bash
# Run commands inside the TeleAvatar sim container (ROS 2 Humble + MuJoCo venv). CPU only, no GPU needed.
#   ~/TeleAvatar2.0/sim.sh                         interactive shell (prompt starts with "(teleavatar)")
#   ~/TeleAvatar2.0/sim.sh python3 convert_urdf.py --check-only --verbose
# Commands run in your current directory: cd ~/TeleAvatar2.0/mujoco first to use the README commands as written.
set -Eeuo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
if [[ ! -f $TA_SIF || ! -x $TA_VENV/bin/python ]]; then
  echo "sim.sh: container or venv missing; run: sbatch $TA_DIR/setup/setup_env.sbatch" >&2
  exit 1
fi

# Same ROS isolation rules as the vendor scripts: default domain 90, never production domain 29, localhost only.
domain=${SIM_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-90}}
if [[ $domain == 29 ]]; then
  echo "sim.sh refuses ROS domain 29 (the production robot); set SIM_ROS_DOMAIN_ID to another domain" >&2
  exit 2
fi

# --cleanenv keeps host settings (modules, PYTHONPATH, LD_PRELOAD, VK_ICD_FILENAMES, ...) out of the container.
# Only the variables below are passed in (SINGULARITYENV_X becomes X inside).
export SINGULARITYENV_TA_DIR=$TA_DIR SINGULARITYENV_TA_VENV=$TA_VENV SINGULARITYENV_UV_CACHE_DIR=$UV_CACHE_DIR
export SINGULARITYENV_SIM_ROS_DOMAIN_ID=$domain SINGULARITYENV_ROS_DOMAIN_ID=$domain SINGULARITYENV_ROS_LOCALHOST_ONLY=1
# Python bytecode goes to ~/.cache instead of __pycache__ folders: the vendor's shipped .pyc files are never loaded
# (only the .py sources run), and the model folder stays as delivered.
export SINGULARITYENV_PYTHONPYCACHEPREFIX=$HOME/.cache/teleavatar_pycache
export SINGULARITYENV_TERM=${TERM:-xterm}
# The image defaults to UTC; use the node's timezone so log timestamps match local time.
tz=${TZ:-$(readlink -f /etc/localtime 2>/dev/null | sed -n 's|^/usr/share/zoneinfo/||p')}
[[ -n $tz ]] && export SINGULARITYENV_TZ=$tz
for var in DISPLAY XAUTHORITY MUJOCO_GL MUJOCO_EGL_DEVICE_ID LP_NUM_THREADS SLURM_CPUS_PER_TASK SLURM_JOB_ID; do
  [[ -n ${!var:-} ]] && export "SINGULARITYENV_$var=${!var}"
done

if (( $# == 0 )); then
  exec singularity exec --cleanenv "$TA_SIF" bash --rcfile "$TA_DIR/container_rc.sh" -i
fi
exec singularity exec --cleanenv "$TA_SIF" bash -c 'source "$TA_DIR/container_rc.sh" && exec "$@"' bash "$@"
