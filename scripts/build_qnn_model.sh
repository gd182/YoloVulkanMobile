#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="vulkansight-qnn:ubuntu24.04"
IMAGE_X86="vulkansight-qnn-x86:ubuntu24.04-v2"
CACHE="${VULKANSIGHT_CACHE:-$HOME/.cache/vulkansight-qnn}"

MODEL=""
ARCH=""
IMGSZ=640
PRECISION="fp16"
CALIB=""
CALIB_COUNT=200
NAME=""
OUT="$ROOT/app/src/main/assets"
SDK=""
FLOAT_IO=0
OPEN_SHELL=0
ASSUME_YES=0
CPUS=4
MEMORY=8
PREPARE=""
ADB=""

usage() {
    cat <<'EOF'
Usage: scripts/build_qnn_model.sh --model best.pt|best.onnx [options]

Builds a QNN HTP context binary for the app (backend "qnn" in models.json).
Everything except the QAIRT SDK is downloaded automatically on first run.

  --model PATH       Ultralytics .pt, or a static-shape .onnx exported without NMS  (required)
  --arch vNN         HTP arch: v68 v69 v73 v75 v79 v81   (default: from the connected phone, else v81)
  --imgsz N          input size used when exporting a .pt                            (default 640)
  --precision P      fp16 (default, no data needed) | int8 (needs --calib)
  --calib DIR        folder with jpg/png images for int8 calibration
  --calib-count N    number of calibration images                                    (default 200)
  --float-io         keep float32 input as well (output is always kept float32)
  --name NAME        output file name without .bin                                   (default <model>_<arch>)
  --out DIR          where to copy the .bin                       (default app/src/main/assets)
  --sdk DIR          QAIRT SDK root (default: local.properties qnn.sdk.dir, $QNN_SDK_ROOT,
                     ~/Library/Qualcomm/AIStack/QAIRT/<latest>)
  --prepare MODE     device | x86 - where the HTP context binary is prepared         (default: device
                     when a phone is connected, else x86; x86 needs Rosetta or a Linux x86_64 host)
  --cpus N           container CPUs when Colima is started                           (default 4)
  --memory GB        container memory when Colima is started                         (default 8)
  --shell            open a shell in the prepared container instead of converting
  -y, --yes          install missing prerequisites without asking
  -h, --help         show this help

The SDK version used here must equal `qnn` in gradle/libs.versions.toml.
EOF
}

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

confirm() {
    [ "$ASSUME_YES" = 1 ] && return 0
    local answer
    read -r -p "$1 [y/N] " answer
    [ "$answer" = y ] || [ "$answer" = Y ]
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        --calib) CALIB="$2"; shift 2 ;;
        --calib-count) CALIB_COUNT="$2"; shift 2 ;;
        --float-io) FLOAT_IO=1; shift ;;
        --name) NAME="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --sdk) SDK="$2"; shift 2 ;;
        --prepare) PREPARE="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --shell) OPEN_SHELL=1; shift ;;
        -y|--yes) ASSUME_YES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

case "$PRECISION" in fp16|int8) ;; *) die "--precision must be fp16 or int8" ;; esac
case "$PREPARE" in ""|device|x86) ;; *) die "--prepare must be device or x86" ;; esac
[ "$OPEN_SHELL" = 1 ] || [ -n "$MODEL" ] || { usage >&2; die "--model is required"; }
[ "$OPEN_SHELL" = 1 ] || [ -f "$MODEL" ] || die "model not found: $MODEL"
if [ "$PRECISION" = int8 ]; then
    [ -d "$CALIB" ] || die "--precision int8 needs --calib DIR with images"
fi

local_property() {
    grep -E "^$1=" "$ROOT/local.properties" 2>/dev/null | head -1 | cut -d= -f2- || true
}

resolve_sdk() {
    local candidate latest
    latest="$(ls -d "$HOME"/Library/Qualcomm/AIStack/QAIRT/* 2>/dev/null | sort -V | tail -1 || true)"
    for candidate in "$SDK" "$(local_property qnn.sdk.dir)" "${QNN_SDK_ROOT:-}" "${QAIRT_SDK_ROOT:-}" "$latest"; do
        if [ -n "$candidate" ] && [ -f "$candidate/include/QNN/QnnInterface.h" ]; then
            SDK="$(cd "$candidate" && pwd)"
            return 0
        fi
    done
    die "QAIRT SDK not found. Pass --sdk DIR or set qnn.sdk.dir in local.properties"
}

find_adb() {
    [ -n "$ADB" ] && return 0
    ADB="$(command -v adb || true)"
    if [ -z "$ADB" ] && [ -x "$HOME/Library/Android/sdk/platform-tools/adb" ]; then
        ADB="$HOME/Library/Android/sdk/platform-tools/adb"
    fi
    return 0
}

device_connected() {
    find_adb
    [ -n "$ADB" ] && "$ADB" devices 2>/dev/null | awk 'NR>1 && $2=="device"' | grep -q .
}

detect_arch() {
    local soc=""
    if device_connected; then
        soc="$("$ADB" shell getprop ro.soc.model 2>/dev/null | tr -d '\r' || true)"
    fi
    if [ -z "$ARCH" ]; then
        case "$soc" in
            SM8350*|SM7350*) ARCH=v68 ;;
            SM8450*) ARCH=v69 ;;
            SM8550*|SM7675*) ARCH=v73 ;;
            SM8650*) ARCH=v75 ;;
            SM8750*) ARCH=v79 ;;
            SM8850*) ARCH=v81 ;;
            *) ARCH=v81; warn "HTP arch not detected (soc='${soc:-no device}'), using v81. Override with --arch" ;;
        esac
    fi
    say "HTP arch: $ARCH (soc: ${soc:-unknown})"
    if [ -z "$PREPARE" ]; then
        if device_connected; then PREPARE=device; else PREPARE=x86; fi
    fi
    say "context binary is prepared on: $PREPARE"
}

ensure_runtime() {
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        return 0
    fi
    if command -v colima >/dev/null 2>&1 && command -v docker >/dev/null 2>&1; then
        say "starting Colima"
        colima start --vm-type vz --cpu "$CPUS" --memory "$MEMORY" --disk 40
        return 0
    fi
    if [ "$(uname -s)" = Darwin ] && command -v brew >/dev/null 2>&1; then
        confirm "Docker is not installed. Install Colima + Docker CLI with Homebrew?" \
            || die "a container runtime is required"
        say "installing Colima and Docker CLI"
        brew install colima docker
        say "starting Colima (Virtualization.framework, native arm64)"
        colima start --vm-type vz --cpu "$CPUS" --memory "$MEMORY" --disk 40
        docker info >/dev/null 2>&1 || die "Docker is still not reachable after starting Colima"
        return 0
    fi
    die "Docker not found. Install Docker Desktop, OrbStack or Colima and retry"
}

check_amd64_runtime() {
    [ "$PRECISION" = int8 ] || return 0
    [ "$(uname -s)" = Darwin ] || return 0
    [ "$(uname -m)" = arm64 ] || return 0

    local context config
    context="$(docker context show 2>/dev/null || true)"
    config="$HOME/.colima/${context#colima-}/colima.yaml"
    [ "$context" = colima ] && config="$HOME/.colima/default/colima.yaml"
    if [ -f "$config" ] && grep -Eq '^rosetta:[[:space:]]*false' "$config"; then
        die "INT8 QAIRT tools cannot run through QEMU on this Mac (hogl ring mutex error). Install Rosetta and restart Colima with: softwareupdate --install-rosetta --agree-to-license; colima stop; colima start --vm-type vz --vz-rosetta --cpu $CPUS --memory $MEMORY"
    fi
}

ensure_image() {
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        say "building the converter image (one time)"
        docker build -t "$IMAGE" - <<'EOF'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip python3-dev \
      build-essential clang git curl ca-certificates libgl1 libglib2.0-0 libtinfo6 zlib1g \
 && rm -rf /var/lib/apt/lists/*
EOF
    fi
    if ! docker image inspect "$IMAGE_X86" >/dev/null 2>&1 ||
       [ "$(docker image inspect -f '{{.Architecture}}' "$IMAGE_X86" 2>/dev/null || true)" != amd64 ]; then
        docker buildx version >/dev/null 2>&1 || die "building the amd64 tools image on an ARM Mac requires Docker buildx. Install it with: brew install docker-buildx; then add /opt/homebrew/lib/docker/cli-plugins to cliPluginsExtraDirs in ~/.docker/config.json"
        say "building the x86_64 QAIRT tools image (one time)"
        docker buildx build --platform linux/amd64 --load -t "$IMAGE_X86" - <<'EOF'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip python3-dev \
      build-essential clang git curl ca-certificates libgl1 libglib2.0-0 libtinfo6 zlib1g \
      libc++1 libc++abi1 libunwind8 libatomic1 \
 && rm -rf /var/lib/apt/lists/*
EOF
    fi
    [ "$(docker image inspect -f '{{.Architecture}}' "$IMAGE_X86")" = amd64 ] \
        || die "$IMAGE_X86 is not an amd64 image; run: docker rmi $IMAGE_X86 and retry"
}

INNER_SCRIPT='
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export QNN_SDK_ROOT=/sdk QAIRT_SDK_ROOT=/sdk
if [ "$(uname -m)" = aarch64 ]; then TOOLS=aarch64-ubuntu-gcc9.4; else TOOLS=x86_64-linux-clang; fi
export PYTHONPATH=/sdk/lib/python
export PATH=/sdk/bin/$TOOLS:/sdk/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=/sdk/lib/$TOOLS
LIB=/sdk/lib/$TOOLS

say() { printf "\033[1m--> %s\033[0m\n" "$*"; }

VENV=/cache/venv-$(uname -m)
if [ ! -x "$VENV/bin/python" ]; then
    say "creating python environment"
    python3 -m venv "$VENV"
fi
. "$VENV/bin/activate"
if [ ! -f "$VENV/.qairt-ok" ]; then
    say "installing QAIRT python dependencies (one time)"
    python3 /sdk/bin/check-python-dependency
    touch "$VENV/.qairt-ok"
fi
if [ ! -f "$VENV/.onnx-pinned" ]; then
    say "pinning onnx to the version the SDK was built with"
    pip install --quiet "onnx==1.16.1" "protobuf==4.25.3" "numpy==1.26.4"
    touch "$VENV/.onnx-pinned"
fi

if [ "${OPEN_SHELL:-0}" = 1 ]; then
    cd /work
    exec bash
fi

cd /work
case "$MODEL_FILE" in
    *.pt)
        EXPORT_VENV=/cache/venv-export-$(uname -m)
        if [ ! -x "$EXPORT_VENV/bin/yolo" ]; then
            say "installing ultralytics (one time)"
            python3 -m venv "$EXPORT_VENV"
            "$EXPORT_VENV/bin/pip" install --quiet ultralytics onnx onnxslim
        fi
        say "exporting ONNX (imgsz=$IMGSZ)"
        cp "/work/in/$MODEL_FILE" "/work/$NAME.pt"
        "$EXPORT_VENV/bin/yolo" export model="/work/$NAME.pt" format=onnx imgsz="$IMGSZ" opset=17 simplify=True
        ONNX="/work/$NAME.onnx"
        ;;
    *.onnx)
        cp "/work/in/$MODEL_FILE" "/work/$NAME.onnx"
        ONNX="/work/$NAME.onnx"
        ;;
    *) echo "unsupported model type: $MODEL_FILE" >&2; exit 1 ;;
esac

if [ "$PRECISION" = int8 ]; then
    say "splitting YOLO boxes and scores before quantization"
    python3 - "$ONNX" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper, shape_inference

path = sys.argv[1]
model = shape_inference.infer_shapes(onnx.load(path))
graph = model.graph
if len(graph.output) != 1:
    raise SystemExit(f"expected one YOLO output, got {len(graph.output)}")

producer = next((node for node in graph.node if graph.output[0].name in node.output), None)
if producer is None or producer.op_type != "Concat" or len(producer.input) != 2:
    raise SystemExit("expected final YOLO Concat(boxes, sigmoid scores)")

axis = next((attr.i for attr in producer.attribute if attr.name == "axis"), None)
if axis not in (1, 2):
    raise SystemExit(f"unsupported YOLO output Concat axis: {axis}")

dims = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
if len(dims) != 3 or dims[axis] <= 4:
    raise SystemExit(f"unsupported YOLO output shape: {dims}")

box_dims = list(dims)
score_dims = list(dims)
box_dims[axis] = 4
score_dims[axis] -= 4
graph.node.extend([
    helper.make_node("Identity", [producer.input[0]], ["qnn_boxes"], name="QnnBoxesOutput"),
    helper.make_node("Identity", [producer.input[1]], ["qnn_scores"], name="QnnScoresOutput"),
])
del graph.output[:]
graph.output.extend([
    helper.make_tensor_value_info("qnn_boxes", TensorProto.FLOAT, box_dims),
    helper.make_tensor_value_info("qnn_scores", TensorProto.FLOAT, score_dims),
])
onnx.checker.check_model(model)
onnx.save(model, path)
print(f"split output into qnn_boxes {box_dims} and qnn_scores {score_dims}")
PY
fi

read -r IN_NAME IN_DIMS OUT_NAME < <(python3 - "$ONNX" "$IMGSZ" <<'PY'
import sys, onnx
m = onnx.load(sys.argv[1])
i = m.graph.input[0]
o = m.graph.output[0]
dims = [d.dim_value if d.dim_value > 0 else None for d in i.type.tensor_type.shape.dim]
size = int(sys.argv[2])
if len(dims) != 4 or None in dims:
    dims = [1, 3, size, size]
print(i.name, ",".join(str(d) for d in dims), o.name)
PY
)
say "input: $IN_NAME [$IN_DIMS]"
if [ "$PRECISION" = int8 ]; then
    say "outputs: qnn_boxes + qnn_scores (kept float32 at graph boundary)"
else
    say "output: $OUT_NAME"
fi

CONVERT_ARGS=(--input_network "$ONNX" --output_path "/work/$NAME.dlc" --source_model_input_shape "$IN_NAME" "$IN_DIMS")
if [ "$PRECISION" = fp16 ]; then
    CONVERT_ARGS+=(--float_bitwidth 16 --float_bias_bitwidth 16)
fi
say "qairt-converter ($PRECISION graph)"
qairt-converter "${CONVERT_ARGS[@]}"
DLC="/work/$NAME.dlc"

if [ "$PRECISION" = int8 ]; then
    say "preparing calibration data ($CALIB_COUNT images)"
    pip install --quiet pillow numpy
    mkdir -p /work/calib
    python3 - "$IN_DIMS" "$CALIB_COUNT" <<'PY'
import glob, sys
import numpy as np
from PIL import Image
n, c, h, w = [int(x) for x in sys.argv[1].split(",")]
files = sorted(glob.glob("/work/calib_src/*.jpg") + glob.glob("/work/calib_src/*.jpeg") + glob.glob("/work/calib_src/*.png"))[: int(sys.argv[2])]
if not files:
    raise SystemExit("no images found in the calibration folder")
with open("/work/calib/list.txt", "w") as lst:
    for k, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        r = min(w / im.width, h / im.height)
        nw, nh = round(im.width * r), round(im.height * r)
        canvas = Image.new("RGB", (w, h), (114, 114, 114))
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
        arr = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        path = f"/work/calib/{k}.raw"
        arr.astype(np.float32).tofile(path)
        lst.write(path + "\n")
print(len(files), "calibration images")
PY
    QUANT_ARGS=(--input_dlc "$DLC" --output_dlc "/work/${NAME}_int8.dlc" --input_list /work/calib/list.txt --act_bitwidth 8 --weights_bitwidth 8)
    if [ "$FLOAT_IO" = 1 ]; then
        QUANT_ARGS+=(--preserve_io_datatype)
    else
        QUANT_ARGS+=(--preserve_io_datatype qnn_boxes qnn_scores)
    fi
    say "qairt-quantizer (int8)"
    qairt-quantizer "${QUANT_ARGS[@]}"
    DLC="/work/${NAME}_int8.dlc"
fi

qairt-dlc-info -i "$DLC" > "/work/$NAME.dlc-info.txt" 2>&1 || true

if [ "$PRECISION" = int8 ]; then
    grep -q "act_bitwidth=8" "/work/$NAME.dlc-info.txt" \
        || { echo "INT8 verification failed: act_bitwidth=8 not found" >&2; exit 1; }
    grep -q "weights_bitwidth=8" "/work/$NAME.dlc-info.txt" \
        || { echo "INT8 verification failed: weights_bitwidth=8 not found" >&2; exit 1; }
    grep -Eq "data type: [us]Fxp_8" "/work/$NAME.dlc-info.txt" \
        || { echo "INT8 verification failed: no 8-bit fixed-point tensors found" >&2; exit 1; }
    say "verified: INT8 weights and activations are present in the DLC"
fi

GRAPH="$(grep -oiE "graph[ _]?name[^A-Za-z0-9_]+[A-Za-z0-9_.:-]+" "/work/$NAME.dlc-info.txt" | head -1 | grep -oE "[A-Za-z0-9_.:-]+$" || true)"
if [ -n "$GRAPH" ]; then
    GRAPHS="\"graphs\": [{\"graph_names\": [\"$GRAPH\"], \"O\": 3}],"
    say "graph: $GRAPH"
else
    GRAPHS=""
fi
cat > /work/htp_ext.json <<JSON
{
  $GRAPHS
  "devices": [{ "dsp_arch": "$ARCH" }]
}
JSON
cat > /work/ctx_config.json <<JSON
{
  "backend_extensions": {
    "shared_library_path": "/sdk/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
    "config_file_path": "/work/htp_ext.json"
  }
}
JSON
echo "$DLC" > /work/dlc.path
say "graph converted: $DLC"
'

STAGE2_SCRIPT='
set -euo pipefail
export LD_LIBRARY_PATH=/sdk/lib/x86_64-linux-clang
LIB=/sdk/lib/x86_64-linux-clang
mkdir -p /work/out
echo "--> qnn-context-binary-generator (HTP $ARCH, x86_64 offline prepare)"
/sdk/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --backend "$LIB/libQnnHtp.so" \
    --model "$LIB/libQnnModelDlc.so" \
    --dlc_path "$(cat /work/dlc.path)" \
    --binary_file "$NAME" \
    --output_dir /work/out \
    --config_file /work/ctx_config.json
test -s "/work/out/$NAME.bin"
echo "--> done: /work/out/$NAME.bin"
'

run_container() {
    local model_file mounts converter_image
    mkdir -p "$CACHE" "$WORK/in"
    mounts=(-v "$SDK:/sdk:ro" -v "$CACHE:/cache" -v "$WORK:/work")
    if [ -n "$MODEL" ]; then
        model_file="$(basename "$MODEL")"
        cp "$MODEL" "$WORK/in/$model_file"
    else
        model_file=""
    fi
    if [ -n "$CALIB" ]; then
        rm -rf "$WORK/calib_src"
        mkdir -p "$WORK/calib_src"
        find "$CALIB" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
            | head -n "$CALIB_COUNT" | while IFS= read -r img; do cp "$img" "$WORK/calib_src/"; done
    fi

    local tty=()
    [ -t 0 ] && [ -t 1 ] && tty=(-it)

    converter_image="$IMAGE"
    local platform=()
    if [ "$PRECISION" = int8 ]; then
        converter_image="$IMAGE_X86"
        platform=(--platform linux/amd64)
        say "running INT8 conversion and calibration on x86_64"
    fi

    docker run --rm ${tty[@]+"${tty[@]}"} "${platform[@]}" "${mounts[@]}" \
        -e MODEL_FILE="$model_file" -e NAME="$NAME" -e ARCH="$ARCH" -e IMGSZ="$IMGSZ" \
        -e PRECISION="$PRECISION" -e CALIB_COUNT="$CALIB_COUNT" -e FLOAT_IO="$FLOAT_IO" \
        -e OPEN_SHELL="$OPEN_SHELL" \
        "$converter_image" bash -c "$INNER_SCRIPT"

    [ "$OPEN_SHELL" = 1 ] && return 0

    if [ "$PREPARE" = device ]; then
        prepare_on_device
    else
        prepare_on_x86 || die "x86_64 preparation failed. QEMU cannot run this tool (hogl mutex error): connect the phone and use --prepare device, or install Rosetta (softwareupdate --install-rosetta) and start Colima with --vz-rosetta, or run on a Linux x86_64 host"
    fi
}

prepare_on_x86() {
    docker run --rm --platform linux/amd64 -v "$SDK:/sdk:ro" -v "$WORK:/work" \
        -e NAME="$NAME" -e ARCH="$ARCH" \
        "$IMAGE_X86" bash -c "$STAGE2_SCRIPT"
}

prepare_on_device() {
    device_connected || die "no phone connected (adb devices)"
    local ver="${ARCH#v}" dir=/data/local/tmp/vulkansight_qnn
    local libs="$SDK/lib/aarch64-android" skel="$SDK/lib/hexagon-$ARCH/unsigned/libQnnHtpV${ARCH#v}Skel.so"
    local dlc file
    dlc="$(cat "$WORK/dlc.path")"
    file="$WORK/$(basename "$dlc")"
    [ -f "$skel" ] || die "no HTP skel for $ARCH in the SDK: $skel"
    [ -f "$file" ] || die "converted graph not found: $file"

    say "pushing the SDK tools and the graph to the phone"
    "$ADB" shell "rm -rf $dir && mkdir -p $dir/out" >/dev/null
    "$ADB" push "$SDK/bin/aarch64-android/qnn-context-binary-generator" "$dir/" >/dev/null
    local lib
    for lib in libQnnHtp.so libQnnHtpPrepare.so "libQnnHtpV${ver}Stub.so" libQnnModelDlc.so libQnnSystem.so libQnnHtpNetRunExtensions.so; do
        "$ADB" push "$libs/$lib" "$dir/" >/dev/null
    done
    "$ADB" push "$skel" "$dir/" >/dev/null
    "$ADB" push "$file" "$dir/model.dlc" >/dev/null
    "$ADB" push "$WORK/htp_ext.json" "$dir/htp_ext.json" >/dev/null
    printf '{\n  "backend_extensions": {\n    "shared_library_path": "%s/libQnnHtpNetRunExtensions.so",\n    "config_file_path": "%s/htp_ext.json"\n  }\n}\n' "$dir" "$dir" \
        > "$WORK/ctx_config.device.json"
    "$ADB" push "$WORK/ctx_config.device.json" "$dir/ctx_config.json" >/dev/null

    say "preparing the context binary on the phone (HTP $ARCH)"
    "$ADB" shell "cd $dir && chmod +x qnn-context-binary-generator \
        && export ADSP_LIBRARY_PATH='$dir;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
        && export LD_LIBRARY_PATH='$dir:/vendor/dsp/cdsp:/vendor/lib64' \
        && ./qnn-context-binary-generator --backend libQnnHtp.so --model libQnnModelDlc.so \
            --dlc_path model.dlc --binary_file $NAME --output_dir out --config_file ctx_config.json" \
        || die "on-device preparation failed (see the output above)"

    mkdir -p "$WORK/out"
    "$ADB" pull "$dir/out/$NAME.bin" "$WORK/out/$NAME.bin" >/dev/null || die "could not pull $NAME.bin from the phone"
    "$ADB" shell "rm -rf $dir" >/dev/null
    test -s "$WORK/out/$NAME.bin" || die "the phone produced an empty context binary"
}

resolve_sdk
say "QAIRT SDK: $SDK"
detect_arch

if [ -z "$NAME" ]; then
    base="$(basename "${MODEL:-model}")"
    NAME="$(printf '%s' "${base%.*}" | tr -c 'A-Za-z0-9_' '_')_${ARCH}"
fi
WORK="$ROOT/build/qnn/$NAME"

ensure_runtime
check_amd64_runtime
ensure_image
run_container

[ "$OPEN_SHELL" = 1 ] && exit 0

mkdir -p "$OUT"
cp "$WORK/out/$NAME.bin" "$OUT/$NAME.bin"
say "wrote $OUT/$NAME.bin ($(du -h "$OUT/$NAME.bin" | cut -f1))"

cat <<EOF

Add to app/src/main/assets/models.json (labels = your classes, in training order):

    {
      "id": "qnn",
      "displayName": "NPU (QNN)",
      "backend": "qnn",
      "model": "$NAME.bin",
      "boxesNormalized": false,
      "confThreshold": 0.3,
      "nmsThreshold": 0.45,
      "labels": ["class0", "class1"]
    }

Then rebuild and install the app. Details: docs/QNN.md
EOF
