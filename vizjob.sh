# vizjob hook for EinsteinVision — `vizjob run <job> [args]` from anywhere in the repo.
VJ_PROJECT=einstein_vision
VJ_HOST=lablaptop-wg                                   # RTX 5060 + Blender 4.5
VJ_DIR='~/Documents/Spring_26/CV/einstein_vision'      # data (P3Data, JSONs) lives there

# local → host: code only (P3Data / JSONs are already on the host)
vj_sync() {
  rsync -a einsteinvision vizjob.sh "$VJ_HOST:${VJ_DIR#\~/}/"
}

_ref_frame() {  # scene frame → mockups/ref/sceneN_front_F.png (real camera)
  local scene=$1 f=$2 out="mockups/ref/scene${1}_front_${2}.png"
  [[ -f "$out" ]] && { echo "$out"; return; }
  mkdir -p mockups/ref
  local v; v=$(ls P3Data/Sequences/scene$scene/Undist/*-front_undistort.mp4 | head -1)
  ffmpeg -loglevel error -y -i "$v" -vf "select=eq(n\,$f)" -vframes 1 "$out" && echo "$out"
}

# Style mockups of single frames, front chase view.
#   vizjob run mockup                  → style C (cinematic), scene1, frames 2131 1530 1200
#   vizjob run mockup -- B 1 600 900   → style B, scene1, frames 600 900
job_mockup() {
  local style=${1:-C} scene=${2:-1}; shift 2 2>/dev/null || shift $#
  local frames=("$@"); [[ ${#frames[@]} -gt 0 ]] || frames=(2131 1530 1200)
  local out="mockups/v2_style${style}"; mkdir -p "$out"
  local sheet=()
  for f in "${frames[@]}"; do
    echo "=== style $style scene $scene frame $f ==="
    local extra=()
    [[ -f road/scene$scene/road_model.json ]] && extra+=(--road road/scene$scene/road_model.json)
    [[ -f road/scene$scene/traffic_lights.json ]] && extra+=(--tl road/scene$scene/traffic_lights.json)
    blender -b --factory-startup --python einsteinvision/mockup_render.py -- \
      --json "phase2_output/scene$scene/detections.json" --frame "$f" --assets P3Data/Assets \
      --style "$style" --out "$out/s${scene}_f${f}.png" "${extra[@]}" 2>&1 | grep -E "^\[mockup\]|^   |Error|Traceback" || return 1
    ref=$(_ref_frame "$scene" "$f") || return 1
    python3 einsteinvision/mockup_compose.py pair "$out/s${scene}_f${f}.png" "$ref" \
      "scene$scene · frame $f · style $style" "$out/pair_s${scene}_f${f}.png"
    vj-post image "$out/pair_s${scene}_f${f}.png" "style $style · scene$scene frame $f (camera | render)"
    sheet+=("$out/pair_s${scene}_f${f}.png")
  done
}

# Road model (BEV lanes + curvature) and traffic-light states for scenes.
#   vizjob run road -- 1 3 4        (debug overlays for 3 frames per scene are posted)
job_road() {
  local scenes=("$@"); [[ ${#scenes[@]} -gt 0 ]] || scenes=(1 3)
  for s in "${scenes[@]}"; do
    local v; v=$(ls P3Data/Sequences/scene$s/Undist/*-front_undistort.mp4 2>/dev/null | head -1)
    [[ -n "$v" ]] || { echo "scene$s: no video, skip"; continue; }
    local n; n=$(python3 -c "import cv2;print(int(cv2.VideoCapture('$v').get(7)))")
    python3 einsteinvision/lane_bev.py --video "$v" --out road/scene$s --post \
      --debug-frames $((n/4)) $((n/2)) $((3*n/4)) || return 1
    python3 einsteinvision/tl_state.py --scene "$s" || return 1
    python3 einsteinvision/tl_strip.py --scene "$s" --out road/scene$s/tl_strip.png && \
      vj-post image road/scene$s/tl_strip.png "scene$s traffic-light crops → classified state (please sanity-check red vs yellow)"
  done
}

# Mockups across scenes: "scene:frame" pairs, style C.
#   vizjob run showcase -- 1:2131 1:1530 3:751 3:2045
job_showcase() {
  for sf in "$@"; do job_mockup C "${sf%%:*}" "${sf##*:}" || return 1; done
}

# Smoothed sequence video, camera | render side by side.
#   vizjob run video -- 1 1900 2140 [samples] [extra sequence_render args]
job_video() {
  local scene=${1:-1} start=${2:-1900} end=${3:-2140} samples=${4:-48}; shift 4 2>/dev/null || shift $#
  local out="renders/scene${scene}_${start}_${end}"
  rm -rf "$out"; mkdir -p "$out"
  blender -b --factory-startup --python einsteinvision/sequence_render.py -- \
    --scene "$scene" --start "$start" --end "$end" --out "$out" --samples "$samples" "$@" 2>&1 \
    | grep --line-buffered -E "^\[seq\]|Error|Traceback|line [0-9]+"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "blender failed"; return 1; }
  grep -q "^" <(ls "$out"/frame_*.png 2>/dev/null) || return 1
  local nf; nf=$(ls "$out"/frame_*.png | wc -l)
  [[ $nf -ge $((end - start)) ]] || { echo "only $nf frames rendered"; return 1; }
  export VH=${VH:-720}
  local fps; fps=$(python3 -c "import json;print(json.load(open('phase2_output/scene$scene/detections.json'))['fps'])")
  local v; v=$(ls P3Data/Sequences/scene$scene/Undist/*-front_undistort.mp4 | head -1)
  ffmpeg -loglevel error -y -i "$v" -vf "select=between(n\,$start\,$end),setpts=N/FRAME_RATE/TB,scale=960:720" \
    -vsync 0 "$out/cam_%05d.png" || return 1
  ffmpeg -loglevel error -y -framerate "$fps" -start_number 1 -i "$out/cam_%05d.png" \
    -framerate "$fps" -start_number "$start" -i "$out/frame_%05d.png" \
    -filter_complex "[0:v]scale=-2:${VH:-720}[c];[1:v]scale=-2:${VH:-720}[r];[c][r]hstack=inputs=2,format=yuv420p" \
    -c:v libx264 -crf ${CRF:-27} -preset slow -movflags +faststart "$out/scene${scene}_${start}_${end}.mp4" || return 1
  ls -la "$out"/*.mp4
  vj-post image "$out/scene${scene}_${start}_${end}.mp4" \
    "scene$scene frames $start-$end · camera | render (smoothed) · $(cat $out/jitter.json)"
}

# Phase-3 vehicle semantics (parked/moving, brake lights, indicators) → road/sceneN/vehicle_state.json
# + labelled crop contact sheet road/sceneN/vehicle_state_check.png (posted).
#   vizjob run vstate -- 1 3
job_vstate() {
  local scenes=("$@"); [[ ${#scenes[@]} -gt 0 ]] || scenes=(1 3)
  for s in "${scenes[@]}"; do
    [[ -f road/scene$s/road_model.json ]] || { echo "scene$s: no road_model.json (run road first)"; return 1; }
    python3 einsteinvision/vehicle_state.py --scene "$s" 2>&1 | tee "road/scene$s/vehicle_state.log" || return 1
    local summ; summ=$(grep -E "^\[vs\] scene$s: [0-9]+ tracks" "road/scene$s/vehicle_state.log" | sed 's/^\[vs\] //')
    vj-post image "road/scene$s/vehicle_state_check.png" \
      "scene$s vehicle_state check sheet (real crops, cyan=lamp ROIs, label=prediction) · $summ"
  done
}

# Phase-3 render check: stills with vehicle_state visualised (brake lamps / blinking amber / parked
# desaturated), camera | render pairs stacked into renders/phase3/check_sN.png (posted).
#   vizjob run phase3 -- "1:500:87:indicator R" "3:1830:860:brake"      (scene:frame:track:caption)
job_phase3() {
  local specs=("$@")
  [[ ${#specs[@]} -gt 0 ]] || specs=("1:486:87:right indicator (blink on phase)" "1:497:87:right indicator (blink off phase)" "1:990:377:braking" "1:655:337:parked"
                                    "3:60:2:stopped/parked" "3:1662:604:braking" "3:1830:860:braking" "3:2070:959:braking")
  mkdir -p renders/phase3/work
  declare -A pairs
  for sp in "${specs[@]}"; do
    local s=${sp%%:*} rest=${sp#*:}; local f=${rest%%:*}; rest=${rest#*:}; local oid=${rest%%:*} txt="t${rest%%:*} ${rest#*:}"
    local st=$((f - 45)); [[ $st -lt 0 ]] && st=0
    local out=renders/phase3/work/s${s}_f${f}
    rm -rf "$out"; mkdir -p "$out"
    blender -b --factory-startup --python einsteinvision/sequence_render.py -- \
      --scene "$s" --start "$st" --end $((f + 45)) --stills "$f" --out "$out" --samples ${SAMPLES:-64} 2>&1 \
      | grep --line-buffered -E "^\[seq\]|Error|Traceback|line [0-9]+"
    [[ -f "$out/frame_$(printf %05d $f).png" ]] || { echo "no render for s$s f$f"; return 1; }
    local ref; ref=$(_ref_frame "$s" "$f") || return 1
    local fr="$out/frame_$(printf %05d $f)"
    python3 einsteinvision/mockup_compose.py inset "$fr.png" "${fr}_objs.json" "$oid" "${fr}_inset.png" || return 1
    python3 - "$fr" "$oid" "$ref" "$s" "$f" <<'PY'
import json, sys; r = [o for o in json.load(open(sys.argv[1] + "_objs.json")) if o["oid"] == sys.argv[2]]
print("[phase3] focus", sys.argv[2], r[0] if r else "NOT PLACED")
from PIL import Image, ImageDraw            # cyan box on the camera frame = the detection being visualised
d = json.load(open(f"phase2_output/scene{sys.argv[4]}/detections.json"))
fr = next(f for f in d["frames"] if f["frame_index"] == int(sys.argv[5]))
im = Image.open(sys.argv[3]).convert("RGB"); dr = ImageDraw.Draw(im)
for o in fr["objects"]:
    if str(o["object_id"]) == sys.argv[2]:
        dr.rectangle(o["bbox_2d"], outline=(0, 255, 255), width=4)
im.save(sys.argv[1] + "_ref.png")
PY
    python3 einsteinvision/mockup_compose.py pair "${fr}_inset.png" "${fr}_ref.png" \
      "scene$s f$f · $txt" "$out/pair.png" || return 1
    pairs[$s]+="$out/pair.png "
  done
  for s in "${!pairs[@]}"; do
    python3 einsteinvision/mockup_compose.py vstack renders/phase3/check_s$s.png ${pairs[$s]} || return 1
    vj-post image renders/phase3/check_s$s.png "phase3 check scene$s · camera | render (brake=red emissive, amber blink, parked desaturated)"
  done
}
