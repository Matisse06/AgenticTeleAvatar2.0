#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_LOCALHOST_ONLY=1
SIM_DOMAIN="${SIM_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-90}}"
if [[ "$SIM_DOMAIN" == "29" ]]; then
  printf 'run_sim.sh refuses ROS domain 29; set SIM_ROS_DOMAIN_ID to an isolated non-production domain.\n' >&2
  exit 2
fi
export ROS_DOMAIN_ID="$SIM_DOMAIN"
if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi
exec python3 "$ROOT_DIR/ros2_sim_node.py" "$@"
