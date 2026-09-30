#!/usr/bin/env bash
# Checks the live MuJoCo viewer (GLFW + OpenGL, software rendered) without a desktop: starts a virtual X display
# (the node's Xvfb), runs the README's interactive control test against run_sim.sh --viewer, runs play.py, saves
# screenshots, and times CPU rendering. Run it on a CPU node (about 2 minutes):
#   srun -p shared -c 4 --mem 8G -t 15 bash ~/TeleAvatar2.0/setup/viewer_check.sh
#
# Known issue: a viewer stopped from code (the end of play.py --duration, Ctrl-C of run_sim.sh --viewer) usually exits
# with a segfault (139). MuJoCo's passive viewer does not wait for its render thread, and at exit glfw.terminate() runs
# while a slow software-rendered frame is still being drawn. It happens after the scripts' own cleanup, so it is
# harmless; this check treats 139 at shutdown as a pass and only fails early crashes.
set -uo pipefail
out=$HOME/TeleAvatar2.0/outputs/verify
sim=$HOME/TeleAvatar2.0/sim.sh
mkdir -p "$out"
command -v Xvfb >/dev/null || { echo "no Xvfb on $(hostname)"; exit 1; }

# -displayfd picks a free display number (other users share the node) and prints it once Xvfb is ready.
exec 3< <(exec Xvfb -displayfd 1 -screen 0 1600x1200x24 -nolisten tcp +extension GLX 2>"$out/xvfb.log")
xvfb=$!
trap 'kill "$xvfb" 2>/dev/null' EXIT
read -r -t 30 num <&3 || { echo "Xvfb did not start; see $out/xvfb.log"; exit 1; }
export DISPLAY=:$num
echo "Xvfb display $DISPLAY on $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs"
cd "$HOME/TeleAvatar2.0/mujoco"
results=()
screenshot() {  # the bundled ffmpeg can grab an X display
  "$sim" bash -c '"$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")" -loglevel error \
    -f x11grab -video_size 1600x1200 -i "$DISPLAY" -frames:v 1 -y "$0"' "$1"
}
stopped_ok() {  # exit code of a viewer process that we stopped ourselves
  (( $1 == 0 || $1 == 139 ))
}

# 1. README "Interactive control test", with the viewer open: run_sim.sh --viewer, then test_control.py.
"$sim" ./run_sim.sh --viewer > "$out/viewer_run_sim.log" 2>&1 &
simpid=$!
sleep 15  # the viewer takes about 10 s to load the meshes on the CPU
if kill -0 "$simpid" 2>/dev/null; then
  ( sleep 4; screenshot "$out/viewer_run_sim.png" ) &
  shot=$!
  "$sim" python3 test_control.py --reordered-names > "$out/viewer_test_control.log" 2>&1; rc=$?
  wait "$shot"
  if (( rc == 0 )); then results+=("PASS  test_control.py with run_sim.sh --viewer"); else
    results+=("FAIL  test_control.py with run_sim.sh --viewer (exit $rc, see $out/viewer_test_control.log)"); fi
  kill -INT "$simpid"; wait "$simpid"; rc=$?
  if stopped_ok "$rc"; then results+=("PASS  run_sim.sh --viewer ran until stopped (exit $rc)"); else
    results+=("FAIL  run_sim.sh --viewer exit $rc (see $out/viewer_run_sim.log)"); fi
else
  wait "$simpid"; results+=("FAIL  run_sim.sh --viewer died early (exit $?, see $out/viewer_run_sim.log)")
fi

# 2. play.py for 20 s, with a screenshot after 15 s.
start=$(date +%s)
"$sim" python3 play.py --duration 20 > "$out/viewer_play.log" 2>&1 &
play=$!
sleep 15
screenshot "$out/viewer_play.png"
wait "$play"; rc=$?
elapsed=$(( $(date +%s) - start ))
if (( elapsed >= 20 )) && stopped_ok "$rc"; then results+=("PASS  play.py viewer ran ${elapsed} s (exit $rc)"); else
  results+=("FAIL  play.py viewer exit $rc after ${elapsed} s (see $out/viewer_play.log)"); fi

# 3. Software rendering speed (Mesa llvmpipe on the allocated CPUs), which also bounds the live viewer's frame rate.
"$sim" python3 - <<'EOF'
import os, time
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
model = mujoco.MjModel.from_xml_path(os.path.expanduser("~/TeleAvatar2.0/mujoco/scene.xml"))
model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
collision = (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (model.geom_group == 0)
print(f"960x720 CPU rendering, {int(model.mesh_facenum.sum()):,} mesh triangles, "
      f"llvmpipe threads: {os.environ.get('LP_NUM_THREADS', 'all cores')}")
start = time.perf_counter()
renderer = mujoco.Renderer(model, 720, 960)
print(f"  context + mesh upload: {time.perf_counter() - start:.1f} s")
for label, hide_collision, shadows in (("collision meshes drawn too, shadows (viewer default)", False, True),
                                       ("visual meshes only, shadows (render_video default)", True, True),
                                       ("visual meshes only, no shadows (--no-shadows)", True, False)):
    model.geom_group[collision] = 3 if hide_collision else 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
    renderer.update_scene(data, camera="overview"); renderer.render()  # warm-up
    start = time.perf_counter()
    for _ in range(10):
        renderer.update_scene(data, camera="overview"); renderer.render()
    ms = (time.perf_counter() - start) / 10 * 1000
    print(f"  {label:55s} {ms:5.0f} ms/frame = {1000 / ms:4.1f} fps")
renderer.close()
EOF

echo; echo "===== viewer summary"
printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q '^FAIL'
