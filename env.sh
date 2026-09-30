# Paths for the TeleAvatar 2.0 MuJoCo sim. Sourced by sim.sh and setup/setup_env.sbatch.
#
#   home      ~/TeleAvatar2.0          vendor model folder (mujoco/, kept as delivered) + our scripts
#   holylabs  $HL/containers, $HL/envs  ROS 2 container + Python venv (persistent, not backed up)

export HL=/n/holylabs/LABS/hankyang_lab/Lab/mvanschalkwijk
export TA_DIR=$HOME/TeleAvatar2.0
export TA_MODEL_DIR=$TA_DIR/mujoco

# Official ROS 2 Humble desktop image (Ubuntu 22.04, Python 3.10: the stack the vendor scripts target; it also has
# Mesa OpenGL/EGL for the viewer and CPU rendering). Pinned by digest to the osrf/ros:humble-desktop pushed 2026-09-10.
export TA_IMAGE=docker://osrf/ros@sha256:fb07245b32187d74350be25323d8ad2f8ca5c25c325759911a1eff2267a49c1e
export TA_SIF=$HL/containers/ros2-humble-desktop_2026-09-10.sif

# mujoco + imageio venv, built with the container's Python 3.10 (so it only works inside the container, via sim.sh).
export TA_VENV=$HL/envs/teleavatar_ros2

# uv cache on the same filesystem as the envs (hardlinks instead of copies), shared with ~/pi05_maniskill/env.sh
export UV_CACHE_DIR=$HL/.uv_cache
