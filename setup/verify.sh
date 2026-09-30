#!/usr/bin/env bash
# Runs every check from the vendor README, plus a CPU render, inside the container; prints a PASS/FAIL summary.
# Takes about a minute on a CPU node (not on the login node):
#   srun -p shared -c 4 --mem 8G -t 20 ~/TeleAvatar2.0/sim.sh bash ~/TeleAvatar2.0/setup/verify.sh
# Outputs (renders, simulator log) go to ~/TeleAvatar2.0/outputs/verify/.
set -uo pipefail
if [[ -z ${TA_DIR:-} || ! -d /opt/ros/humble ]]; then
  echo "run this through sim.sh (see the usage line at the top)" >&2
  exit 1
fi
out=$TA_DIR/outputs/verify
mkdir -p "$out"
cd "$TA_DIR/mujoco"
echo "node $(hostname), ROS_DOMAIN_ID=$ROS_DOMAIN_ID, $(python3 -c 'import mujoco; print("mujoco", mujoco.__version__)')"

results=()
run() {
  local name=$1; shift
  echo; echo "=== $name: $*"
  if "$@"; then results+=("PASS  $name"); else results+=("FAIL  $name (exit $?)"); fi
}

# README "Generate and test". Regenerating robot.xml (and the unit tests, which rebuild it too) should not change it.
before=$(sha256sum robot.xml)
run "convert_urdf.py --verbose" python3 convert_urdf.py --verbose
run "convert_urdf.py --check-only" python3 convert_urdf.py --check-only --verbose
run "unit tests" python3 -m unittest discover -s tests -v
if [[ $(sha256sum robot.xml) == "$before" ]]; then
  results+=("PASS  robot.xml identical after regeneration")
else
  results+=("NOTE  robot.xml content changed after regeneration")
fi

# README "ROS 2 simulator": the smoke test starts and stops its own simulator.
run "smoke_test.py" python3 smoke_test.py

# README "Interactive control test": test_control.py attaches to an already-running simulator.
./run_sim.sh > "$out/run_sim.log" 2>&1 &
sim=$!
run "test_control.py --reordered-names" python3 test_control.py --reordered-names
kill -TERM "$sim" 2>/dev/null; wait "$sim" 2>/dev/null

# Offscreen CPU rendering (EGL + Mesa llvmpipe), for videos viewable in VS Code.
run "render still" python3 "$TA_DIR/scripts/render_video.py" --output "$out/home.png"
run "render video" python3 "$TA_DIR/scripts/render_video.py" --output "$out/wave.mp4"

echo; echo "===== summary ($(date '+%F %T'))"
printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q '^FAIL'
