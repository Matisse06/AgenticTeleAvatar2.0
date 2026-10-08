#!/usr/bin/env bash
# Runs every check from the vendor README on the vendor model (mujoco/), then the same checks on the new model
# (model/), plus CPU renders of both, then the scenes' tests (scenes/) and a still of one, inside the container; prints
# a PASS/FAIL summary. On a CPU node (not the login node); the new model's video takes most of the time (about 5
# minutes on 4 cores):
#   srun -p shared -c 4 --mem 8G -t 30 ~/TeleAvatar2.0/sim.sh bash ~/TeleAvatar2.0/setup/verify.sh
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
run "render still" python3 "$TA_DIR/scripts/render_video.py" --model "$TA_DIR/mujoco/scene.xml" --output "$out/home.png"
run "render video" python3 "$TA_DIR/scripts/render_video.py" --model "$TA_DIR/mujoco/scene.xml" --output "$out/wave.mp4"

# The new model (model/, see model/README.md). Its meshes come from the vendor archive: setup/unpack_assets.py.
cd "$TA_DIR"
run "new: assets verified" python3 setup/unpack_assets.py --verify-only
generated="model/robot.xml model/robot_mobile.xml model/scene_mobile.xml"
before=$(sha256sum $generated)
run "new: convert.py" python3 model/convert.py
run "new: unit tests" python3 -m unittest discover -s model/tests -v
if [[ $(sha256sum $generated) == "$before" ]]; then
  results+=("PASS  new: robot.xml, robot_mobile.xml, scene_mobile.xml identical after regeneration")
else
  results+=("NOTE  new: generated model files changed after regeneration")
fi
run "new: smoke_test.py" python3 model/smoke_test.py
./model/run_sim.sh > "$out/run_sim_new.log" 2>&1 &
sim=$!
for _ in $(seq 120); do  # wait (up to 60 s) until the simulator has loaded the model
  grep -q "simulator ready" "$out/run_sim_new.log" 2>/dev/null && break
  sleep 0.5
done
run "new: vendor test_control.py --reordered-names" python3 mujoco/test_control.py --reordered-names
kill -TERM "$sim" 2>/dev/null; wait "$sim" 2>/dev/null
run "new: render still" python3 scripts/render_video.py --output "$out/new_home.png"
run "new: render video" python3 scripts/render_video.py --output "$out/new_wave.mp4"

# The scenes (scenes/, see scenes/README.md): the robot at a table with objects.
run "scenes: unit tests" python3 -m unittest discover -s scenes/tests -v
run "scenes: render still" python3 scripts/render_video.py --model scenes/table_cubes.xml --output "$out/table_cubes.png"

echo; echo "===== summary ($(date '+%F %T'))"
printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q '^FAIL'
