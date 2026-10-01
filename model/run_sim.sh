#!/usr/bin/env bash
# Run the new model's ROS 2 simulator (ros2_sim_node.py; options: --viewer, --lift, --gripper, ...). Same isolation
# rules as mujoco/run_sim.sh: localhost-only discovery, ROS domain 90 by default, never the production robot's 29.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_LOCALHOST_ONLY=1                       # Humble
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST    # Jazzy and later (ROS_LOCALHOST_ONLY is deprecated there)
SIM_DOMAIN="${SIM_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-90}}"
if [[ "$SIM_DOMAIN" == "29" ]]; then
  printf 'run_sim.sh refuses ROS domain 29; set SIM_ROS_DOMAIN_ID to an isolated non-production domain.\n' >&2
  exit 2
fi
export ROS_DOMAIN_ID="$SIM_DOMAIN"
if [[ -z "${ROS_DISTRO:-}" ]]; then  # not sourced yet (sim.sh's container sources Humble itself)
  for distro in humble jazzy rolling; do
    if [[ -f /opt/ros/$distro/setup.bash ]]; then
      set +u
      # shellcheck disable=SC1090
      source "/opt/ros/$distro/setup.bash"
      set -u
      break
    fi
  done
fi
exec python3 "$ROOT_DIR/ros2_sim_node.py" "$@"
