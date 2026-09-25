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
  _wait_gpu_free
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

# Stage B1: road + TL + vehicle_state for many scenes; continues past per-scene failures and
# prints a summary table (fails overall if any scene failed, so logs get read).
#   vizjob run b1 -- 2 4 5 6 7 8 9 10 11 12 13
job_b1() {
  local scenes=("$@"); [[ ${#scenes[@]} -gt 0 ]] || scenes=($(seq 1 13))
  local ok=() bad=()
  for s in "${scenes[@]}"; do
    echo "######## scene$s road ########"; local t0=$SECONDS
    if ! job_road "$s"; then bad+=("s$s:road"); continue; fi
    echo "######## scene$s vstate ($((SECONDS - t0))s road) ########"
    if ! job_vstate "$s"; then bad+=("s$s:vstate"); continue; fi
    ok+=("s$s"); echo "######## scene$s done in $((SECONDS - t0))s ########"
  done
  echo "B1 SUMMARY ok=[${ok[*]}] failed=[${bad[*]}]"
  [[ ${#bad[@]} -eq 0 ]]
}

# Wait until no other vizjob tmux session (besides ours) is running, so GPU timings/renders don't overlap.
_wait_gpu_free() {   # wait only for vj- sessions created before ours (FIFO, no deadlock)
  local me; me=$(tmux display-message -p '#S' 2>/dev/null)
  local mt; mt=$(tmux display-message -p '#{session_created}' 2>/dev/null); mt=${mt:-9999999999}
  while tmux ls -F '#{session_name} #{session_created}' 2>/dev/null \
        | awk -v me="$me" -v mt="$mt" '$1 ~ /^vj-/ && $1 != me && $2 < mt' | grep -q .; do sleep 30; done
}

# Stage B2 timing: render N frames of a scene, report s/frame (no video).
#   vizjob run speed -- 1 900 30 [samples] [extra args]
job_speed() {
  local scene=${1:-1} start=${2:-900} n=${3:-30} samples=${4:-32}; shift 4 2>/dev/null || shift $#
  _wait_gpu_free
  local out=renders/speedtest; rm -rf "$out"
  local t0=$SECONDS
  blender -b --factory-startup --python einsteinvision/sequence_render.py -- \
    --scene "$scene" --start "$start" --end $((start + n - 1)) --out "$out" --samples "$samples" --jpeg "$@" 2>&1 \
    | grep --line-buffered -E "^\[seq\]|Error|Traceback"
  local nf; nf=$(ls "$out"/frame_*.jpg 2>/dev/null | wc -l)
  echo "SPEED scene$scene samples=$samples frames=$nf wall=$((SECONDS - t0))s"
  [[ $nf -gt 0 ]] || return 1
  vj-post image "$out/frame_$(printf %05d $((start + n / 2))).jpg" "speedtest s$scene f$((start + n / 2)) samples=$samples ($nf frames, $((SECONDS - t0))s wall)"
}

# Stage B2: full-length camera | render videos → renders/full/sceneN.mp4. Chunked Blender processes,
# JPEG frames deleted after the mp4 is verified. Env: SAMPLES (32) STEP (1) CHUNK (600).
#   vizjob run full -- 1 2 3 ... 13
job_full() {
  local scenes=("$@"); [[ ${#scenes[@]} -gt 0 ]] || scenes=($(seq 1 13))
  local samples=${SAMPLES:-32} step=${STEP:-1} chunk=${CHUNK:-600} ok=() bad=()
  _wait_gpu_free
  mkdir -p renders/full
  for s in "${scenes[@]}"; do
    local free; free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
    if [[ $free -lt 5 ]]; then discord-notify --alert "ev-phase3: laptop disk ${free}G free, stopping job_full" 2>/dev/null; bad+=("s$s:disk"); break; fi
    echo "######## full scene$s ########"; local t0=$SECONDS
    local out=renders/full/work_s$s; rm -rf "$out"; mkdir -p "$out"
    read first last fps < <(python3 -c "
import json; d=json.load(open('phase2_output/scene$s/detections.json')); k=[f['frame_index'] for f in d['frames']]
print(min(k), max(k), d.get('fps', 30))")
    local st=$first fail=0
    while [[ $st -le $last ]]; do
      local en=$((st + chunk - 1)); [[ $en -gt $last ]] && en=$last
      blender -b --factory-startup --python einsteinvision/sequence_render.py -- \
        --scene "$s" --start "$st" --end "$en" --out "$out" --samples "$samples" --step "$step" --jpeg 2>&1 \
        | grep --line-buffered -E "^\[seq\] (done|[0-9]+/[0-9]+ frame [0-9]+ .*(eta)|vstate)|Error|Traceback"
      [[ ${PIPESTATUS[0]} -eq 0 ]] || { fail=1; break; }
      st=$((en + 1))
    done
    local nf; nf=$(ls "$out"/frame_*.jpg 2>/dev/null | wc -l)
    [[ $fail -eq 0 && $nf -gt 0 ]] || { bad+=("s$s:render"); continue; }
    local r0; r0=$(ls "$out"/frame_*.jpg | head -1 | grep -o '[0-9]\{5\}' | sed 's/^0*//'); r0=${r0:-0}
    local v; v=$(ls P3Data/Sequences/scene$s/Undist/*-front_undistort.mp4 | head -1)
    # render frames may be sparse (step / missing road frames): glob at fps/step then resample to fps
    ffmpeg -loglevel error -y -i "$v" -framerate "$(python3 -c "print($fps/$step)")" -pattern_type glob -i "$out/frame_*.jpg" \
      -filter_complex "[0:v]trim=start_frame=$r0,setpts=PTS-STARTPTS,scale=-2:720[c];[1:v]fps=$fps,scale=-2:720[r];[c][r]hstack=inputs=2:shortest=1,format=yuv420p" \
      -r "$fps" -c:v libx264 -crf ${CRF:-27} -preset medium -movflags +faststart "renders/full/scene$s.mp4" \
      || { bad+=("s$s:ffmpeg"); continue; }
    local d sd; d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "renders/full/scene$s.mp4")
    sd=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$v")
    echo "scene$s: $nf frames rendered, video ${d}s vs source ${sd}s, $(( (SECONDS - t0) / 60 )) min"
    if python3 -c "import sys; sys.exit(0 if $d >= 0.9*$sd else 1)"; then
      ok+=("s$s")
      local mid; mid=$(ls "$out"/frame_*.jpg | sed -n "$((nf / 2))p")
      [[ " ${POST:-1 7 13} " == *" $s "* ]] && vj-post image "renders/full/scene$s.mp4" "full scene$s · camera | render · ${d}s"
      rm -rf "$out"
    else bad+=("s$s:short"); fi
  done
  echo "B2 SUMMARY ok=[${ok[*]}] failed=[${bad[*]}]"
  [[ ${#bad[@]} -eq 0 ]]
}

# Stage C: top + chase (EEVEE, one scene build per frame for both views) → 2×2 composite with renders/full/sceneN.mp4
#   ┌ dashcam ┬ front ┐   1920×1080; the full mp4 is already camera|render synced, so cells 1–2 are crops of it
#   └ top     ┴ chase ┘
job_composite() {
  local scenes=("$@"); [[ ${#scenes[@]} -gt 0 ]] || scenes=($(seq 1 13))
  local samples=${SAMPLES:-16} step=${STEP:-2} chunk=${CHUNK:-600} ok=() bad=()
  _wait_gpu_free
  mkdir -p renders/composite
  for s in "${scenes[@]}"; do
    local free; free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
    if [[ $free -lt 5 ]]; then discord-notify --alert "ev-phase3: laptop disk ${free}G free, stopping job_composite" 2>/dev/null; bad+=("s$s:disk"); break; fi
    [[ -f renders/full/scene$s.mp4 ]] || { bad+=("s$s:nofull"); continue; }
    echo "######## composite scene$s ########"; local t0=$SECONDS
    local out=renders/composite/work_s$s; rm -rf "$out"; mkdir -p "$out"
    read first last fps < <(python3 -c "
import json; d=json.load(open('phase2_output/scene$s/detections.json')); k=[f['frame_index'] for f in d['frames']]
print(min(k), max(k), d.get('fps', 30))")
    local st=$first fail=0
    while [[ $st -le $last ]]; do
      local en=$((st + chunk - 1)); [[ $en -gt $last ]] && en=$last
      blender -b --factory-startup --python einsteinvision/sequence_render.py -- \
        --scene "$s" --start "$st" --end "$en" --out "$out" --samples "$samples" --step "$step" --jpeg \
        --view top chase --engine eevee --res 960 540 2>&1 \
        | grep --line-buffered -E "^\[seq\] (done|[0-9]+/[0-9]+ frame [0-9]+ .*(eta)|vstate)|Error|Traceback"
      [[ ${PIPESTATUS[0]} -eq 0 ]] || { fail=1; break; }
      st=$((en + 1))
    done
    local nf; nf=$(ls "$out"/top/frame_*.jpg 2>/dev/null | wc -l)
    [[ $fail -eq 0 && $nf -gt 0 ]] || { bad+=("s$s:render"); continue; }
    local r=$(python3 -c "print($fps/$step)")
    ffmpeg -loglevel error -y -i "renders/full/scene$s.mp4" \
      -framerate "$r" -pattern_type glob -i "$out/top/frame_*.jpg" \
      -framerate "$r" -pattern_type glob -i "$out/chase/frame_*.jpg" \
      -filter_complex "[0:v]split[a][b];[a]crop=iw/2:ih:0:0,scale=960:540,setsar=1[cam];[b]crop=iw/2:ih:iw/2:0,scale=960:540,setsar=1[fr];\
[1:v]fps=$fps,scale=960:540,setsar=1[tp];[2:v]fps=$fps,scale=960:540,setsar=1[ch];\
[cam][fr]hstack[u];[tp][ch]hstack[l];[u][l]vstack=shortest=1,format=yuv420p" \
      -r "$fps" -c:v libx264 -crf ${CRF:-27} -preset medium -movflags +faststart "renders/composite/scene$s.mp4" \
      || { bad+=("s$s:ffmpeg"); continue; }
    local v; v=$(ls P3Data/Sequences/scene$s/Undist/*-front_undistort.mp4 | head -1)
    local d sd; d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "renders/composite/scene$s.mp4")
    sd=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$v")
    echo "scene$s: $nf frames/view, composite ${d}s vs source ${sd}s, $(( (SECONDS - t0) / 60 )) min"
    if python3 -c "import sys; sys.exit(0 if $d >= 0.9*$sd else 1)"; then
      ok+=("s$s")
      [[ " ${POST:-1 7 13} " == *" $s "* ]] && vj-post image "renders/composite/scene$s.mp4" "composite scene$s · dashcam | front / top | chase · ${d}s"
      rm -rf "$out"
    else bad+=("s$s:short"); fi
  done
  echo "C2 SUMMARY ok=[${ok[*]}] failed=[${bad[*]}]"
  [[ ${#bad[@]} -eq 0 ]]
}

# vizjob rsync touches sequence_render.py → phase3 stills must be re-rendered (A3 mtime check) before composites
job_stagec() { job_phase3 && job_composite "$@"; }
