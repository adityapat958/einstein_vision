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
    blender -b --factory-startup --python einsteinvision/mockup_render.py -- \
      --json "phase2_output/scene$scene/detections.json" --frame "$f" --assets P3Data/Assets \
      --style "$style" --out "$out/s${scene}_f${f}.png" 2>&1 | grep -E "^\[mockup\]|^   |Error|Traceback" || return 1
    ref=$(_ref_frame "$scene" "$f") || return 1
    python3 einsteinvision/mockup_compose.py pair "$out/s${scene}_f${f}.png" "$ref" \
      "scene$scene · frame $f · style $style" "$out/pair_s${scene}_f${f}.png"
    vj-post image "$out/pair_s${scene}_f${f}.png" "style $style · scene$scene frame $f (camera | render)"
    sheet+=("$out/pair_s${scene}_f${f}.png")
  done
}
