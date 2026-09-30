#!/bin/bash
# One-time setup of the TeleAvatar 2.0 MuJoCo sim environment (CPU only, no GPU):
#   1. pulls the pinned ROS 2 Humble desktop image ($TA_IMAGE in env.sh) as a Singularity image to $TA_SIF
#   2. creates the venv $TA_VENV with the container's Python 3.10 and installs mujoco + imageio[ffmpeg]
# Usage: mkdir -p ~/TeleAvatar2.0/logs && sbatch ~/TeleAvatar2.0/setup/setup_env.sbatch
# Safe to re-run: an existing image is kept, and the install step only adds what is missing.
#SBATCH --job-name=teleavatar_setup
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/n/home00/mvanschalkwijk/TeleAvatar2.0/logs/%x_%j.log

set -Eeuo pipefail
source "$HOME/TeleAvatar2.0/env.sh"
echo "$(date '+%F %T') node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"

# 1. Container. Layer downloads and the squashfs build use node-local /scratch (fast), deleted when the job ends.
if [[ ! -f $TA_SIF ]]; then
  base=/scratch; [[ -d $base && -w $base ]] || base=/n/netscratch/hankyang_lab/Lab/mvanschalkwijk
  work=$base/$USER/teleavatar_setup_${SLURM_JOB_ID:-$$}
  mkdir -p "$work/cache" "$work/tmp" "$(dirname "$TA_SIF")"
  trap 'rm -rf "$work" "$TA_SIF.partial"; rmdir "$(dirname "$work")" 2>/dev/null || true' EXIT
  export SINGULARITY_CACHEDIR=$work/cache SINGULARITY_TMPDIR=$work/tmp
  echo "pulling $TA_IMAGE (temp files in $work)"
  singularity pull "$TA_SIF.partial" "$TA_IMAGE"
  mv "$TA_SIF.partial" "$TA_SIF"
fi
ls -la "$TA_SIF"

# 2. venv, built inside the container with its /usr/bin/python3 (3.10, what ROS Humble's rclpy is built for).
# --system-site-packages keeps the container's Python packages visible; numpy<2 because ROS Humble's compiled
# message modules are built against numpy 1.x.
export SINGULARITYENV_TA_DIR=$TA_DIR SINGULARITYENV_TA_VENV=$TA_VENV SINGULARITYENV_UV_CACHE_DIR=$UV_CACHE_DIR
export SINGULARITYENV_UV_PYTHON_DOWNLOADS=never
singularity exec --cleanenv "$TA_SIF" bash -c '
  set -Eeuo pipefail
  uv=$HOME/.local/bin/uv
  [[ -x $TA_VENV/bin/python ]] || "$uv" venv --python /usr/bin/python3 --system-site-packages "$TA_VENV"
  "$uv" pip install --python "$TA_VENV/bin/python" "mujoco==3.14.0" "numpy<2" "imageio[ffmpeg]"
  "$uv" pip freeze --python "$TA_VENV/bin/python" > "$TA_DIR/setup/requirements.lock"
  set +u  # ROS setup scripts read unset variables
  source /opt/ros/humble/setup.bash
  source "$TA_VENV/bin/activate"
  python3 - <<EOF
import sys, numpy, mujoco, imageio, imageio_ffmpeg, rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32, String
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "from", numpy.__file__)
print("mujoco", mujoco.__version__, "| imageio", imageio.__version__, "| ffmpeg", imageio_ffmpeg.get_ffmpeg_version())
print("rclpy + sensor_msgs + std_msgs import OK; ROS_DISTRO =", __import__("os").environ.get("ROS_DISTRO"))
EOF
'
echo "$(date '+%F %T') setup done; venv packages pinned in $TA_DIR/setup/requirements.lock"
