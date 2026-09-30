# Sourced inside the container by sim.sh: ROS 2 Humble + the MuJoCo venv.
source /opt/ros/humble/setup.bash
source "$TA_VENV/bin/activate"
export PATH="$PATH:$HOME/.local/bin"   # uv, to add packages later: uv pip install <package>
# Mesa's CPU rasterizer (llvmpipe) otherwise starts a thread per core on the node, not per allocated core.
[[ -n ${SLURM_CPUS_PER_TASK:-} && -z ${LP_NUM_THREADS:-} ]] && export LP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PS1='(teleavatar) \u@\h:\w\$ '
