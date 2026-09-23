#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VULKANSIGHT_CACHE:-$HOME/.cache/vulkansight-qnn}"
MODEL=""
ARCH="v81"
IMGSZ=640
PRECISION="fp16"
CALIB=""
CALIB_COUNT=200
NAME=""
OUT="$ROOT/app/src/main/assets"
SDK="${QNN_SDK_ROOT:-${QAIRT_SDK_ROOT:-}}"
FLOAT_IO=0

usage() {
    cat <<'EOF'
Usage: scripts/build_qnn_model_ubuntu_x86.sh --model best.pt|best.onnx [options]

Native Ubuntu x86_64 builder for a QNN HTP context binary. Docker is not used.

  --model PATH       Ultralytics .pt or static-shape .onnx without NMS       (required)
  --arch vNN         HTP architecture: v68 v69 v73 v75 v79 v81              (default v81)
  --imgsz N          input size for .pt export                               (default 640)
  --precision P      fp16 or int8                                             (default fp16)
  --calib DIR        jpg/jpeg/png calibration images for INT8
  --calib-count N    maximum calibration image count                          (default 200)
  --float-io         preserve float input too; outputs are always float
  --name NAME        output filename without .bin                             (default <model>_<arch>)
  --out DIR          destination directory                    (default app/src/main/assets)
  --sdk DIR          QAIRT SDK root (or set QNN_SDK_ROOT/QAIRT_SDK_ROOT)
  -h, --help         show this help

Host requirements: Ubuntu 22.04/24.04 x86_64, Python 3 with venv, and QAIRT SDK.
The QAIRT version must match the qnn-runtime version used by the Android app.
EOF
}

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command '$1' (install: sudo apt install $2)"; }

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
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

[ "$(uname -s)" = Linux ] || die "this script runs only on Linux; use build_qnn_model.sh on macOS"
[ "$(uname -m)" = x86_64 ] || die "x86_64 host required, got $(uname -m)"
case "$ARCH" in v68|v69|v73|v75|v79|v81) ;; *) die "unsupported --arch: $ARCH" ;; esac
case "$PRECISION" in fp16|int8) ;; *) die "--precision must be fp16 or int8" ;; esac
[ -n "$MODEL" ] || { usage >&2; die "--model is required"; }
[ -f "$MODEL" ] || die "model not found: $MODEL"
if [ "$PRECISION" = int8 ]; then
    [ -d "$CALIB" ] || die "--precision int8 needs --calib DIR with images"
fi

need python3 python3
python3 -m venv --help >/dev/null 2>&1 || die "Python venv is unavailable (install: sudo apt install python3-venv)"

local_property() {
    grep -E "^$1=" "$ROOT/local.properties" 2>/dev/null | head -1 | cut -d= -f2- || true
}

sdk_root_from() {
    local candidate="$1" header root i
    [ -n "$candidate" ] || return 1
    candidate="${candidate/#\~/$HOME}"
    [ -e "$candidate" ] || return 1
    candidate="$(cd "$candidate" 2>/dev/null && pwd)" || return 1

    root="$candidate"
    for i in 0 1 2 3 4; do
        if [ -f "$root/include/QNN/QnnInterface.h" ]; then
            printf '%s\n' "$root"
            return 0
        fi
        [ "$root" != / ] || break
        root="$(dirname "$root")"
    done

    header="$(find "$candidate" -maxdepth 7 -type f -path '*/include/QNN/QnnInterface.h' \
        2>/dev/null | sort -V | tail -1 || true)"
    [ -n "$header" ] || return 1
    printf '%s\n' "${header%/include/QNN/QnnInterface.h}"
}

resolve_sdk() {
    local candidate found search_root header
    for candidate in "$SDK" "$(local_property qnn.sdk.dir)" \
        "${QNN_SDK_ROOT:-}" "${QAIRT_SDK_ROOT:-}"; do
        found="$(sdk_root_from "$candidate" || true)"
        if [ -n "$found" ]; then
            SDK="$found"
            return 0
        fi
    done

    for search_root in \
        "$HOME/Qualcomm" "$HOME/QAIRT" "$HOME/qairt" "$HOME/Downloads" \
        /opt/qcom /opt/qualcomm /usr/local/qualcomm; do
        [ -d "$search_root" ] || continue
        header="$(find "$search_root" -maxdepth 8 -type f -path '*/include/QNN/QnnInterface.h' \
            2>/dev/null | sort -V | tail -1 || true)"
        if [ -n "$header" ]; then
            SDK="${header%/include/QNN/QnnInterface.h}"
            return 0
        fi
    done

    cat >&2 <<EOF
error: QAIRT SDK root was not found.
Expected file: <SDK>/include/QNN/QnnInterface.h

Locate it:
  find \$HOME /opt -type f -path '*/include/QNN/QnnInterface.h' 2>/dev/null

Then pass the directory above 'include':
  --sdk /path/to/QAIRT/2.50.0
or:
  export QNN_SDK_ROOT=/path/to/QAIRT/2.50.0
EOF
    exit 1
}

resolve_sdk
SDK="$(cd "$SDK" && pwd)"
say "QAIRT SDK: $SDK"

TOOLS="$SDK/bin/x86_64-linux-clang"
LIB="$SDK/lib/x86_64-linux-clang"
CONVERTER="$TOOLS/qairt-converter"
QUANTIZER="$TOOLS/qairt-quantizer"
GENERATOR="$TOOLS/qnn-context-binary-generator"
for tool in "$CONVERTER" "$QUANTIZER" "$GENERATOR"; do
    [ -f "$tool" ] || die "SDK tool not found: $tool (the Linux x86_64 QAIRT package is required)"
    [ -x "$tool" ] || die "SDK tool is not executable: $tool (run: chmod +x '$tool')"
done
for lib in libQnnHtp.so libQnnModelDlc.so libQnnHtpNetRunExtensions.so; do
    [ -f "$LIB/$lib" ] || die "SDK library not found: $LIB/$lib"
done

if [ -z "$NAME" ]; then
    base="$(basename "$MODEL")"
    NAME="$(printf '%s' "${base%.*}" | tr -c 'A-Za-z0-9_' '_')_${ARCH}"
fi
WORK="$ROOT/build/qnn/$NAME"
mkdir -p "$CACHE" "$WORK" "$OUT"

export QNN_SDK_ROOT="$SDK" QAIRT_SDK_ROOT="$SDK"
export PYTHONPATH="$SDK/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$TOOLS:$PATH"
export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VENV="$CACHE/venv-ubuntu-x86_64"
if [ ! -x "$VENV/bin/python" ]; then
    say "creating QAIRT Python environment"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
if [ ! -f "$VENV/.qairt-ok" ]; then
    say "installing QAIRT Python dependencies (one time)"
    python3 "$SDK/bin/check-python-dependency"
    python3 -m pip install --quiet "onnx==1.16.1" "protobuf==4.25.3" "numpy==1.26.4" pillow
    touch "$VENV/.qairt-ok"
fi

case "$MODEL" in
    *.pt)
        EXPORT_VENV="$CACHE/venv-export-ubuntu-x86_64"
        if [ ! -x "$EXPORT_VENV/bin/yolo" ]; then
            say "installing Ultralytics exporter (one time)"
            python3 -m venv "$EXPORT_VENV"
            "$EXPORT_VENV/bin/pip" install --quiet ultralytics onnx onnxslim
        fi
        say "exporting ONNX (imgsz=$IMGSZ)"
        cp "$MODEL" "$WORK/$NAME.pt"
        "$EXPORT_VENV/bin/yolo" export model="$WORK/$NAME.pt" format=onnx imgsz="$IMGSZ" opset=17 simplify=True
        ONNX="$WORK/$NAME.onnx"
        ;;
    *.onnx)
        cp "$MODEL" "$WORK/$NAME.onnx"
        ONNX="$WORK/$NAME.onnx"
        ;;
    *) die "unsupported model type: $MODEL" ;;
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
producer = next((n for n in graph.node if graph.output[0].name in n.output), None)
if producer is None or producer.op_type != "Concat" or len(producer.input) != 2:
    raise SystemExit("expected final YOLO Concat(boxes, sigmoid scores)")
axis = next((a.i for a in producer.attribute if a.name == "axis"), None)
if axis not in (1, 2):
    raise SystemExit(f"unsupported YOLO output Concat axis: {axis}")
dims = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
if len(dims) != 3 or dims[axis] <= 4:
    raise SystemExit(f"unsupported YOLO output shape: {dims}")
box_dims, score_dims = list(dims), list(dims)
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
print(f"qnn_boxes {box_dims}; qnn_scores {score_dims}")
PY
fi

read -r IN_NAME IN_DIMS OUT_NAME < <(python3 - "$ONNX" "$IMGSZ" <<'PY'
import sys, onnx
m = onnx.load(sys.argv[1])
i, o = m.graph.input[0], m.graph.output[0]
dims = [d.dim_value if d.dim_value > 0 else None for d in i.type.tensor_type.shape.dim]
if len(dims) != 4 or None in dims:
    dims = [1, 3, int(sys.argv[2]), int(sys.argv[2])]
print(i.name, ",".join(map(str, dims)), o.name)
PY
)
say "input: $IN_NAME [$IN_DIMS]"

CONVERT_ARGS=(--input_network "$ONNX" --output_path "$WORK/$NAME.dlc"
              --source_model_input_shape "$IN_NAME" "$IN_DIMS")
if [ "$PRECISION" = fp16 ]; then
    CONVERT_ARGS+=(--float_bitwidth 16 --float_bias_bitwidth 16)
fi
say "qairt-converter ($PRECISION graph)"
"$CONVERTER" "${CONVERT_ARGS[@]}"
DLC="$WORK/$NAME.dlc"

if [ "$PRECISION" = int8 ]; then
    CALIB="$(cd "$CALIB" && pwd)"
    CALIB_WORK="$WORK/calib"
    mkdir -p "$CALIB_WORK"
    say "preparing calibration data ($CALIB_COUNT maximum)"
    python3 - "$IN_DIMS" "$CALIB_COUNT" "$CALIB" "$CALIB_WORK" <<'PY'
import glob, os, sys
import numpy as np
from PIL import Image

n, c, h, w = [int(x) for x in sys.argv[1].split(",")]
limit, source, output = int(sys.argv[2]), sys.argv[3], sys.argv[4]
files = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    files.extend(glob.glob(os.path.join(source, pattern)))
files = sorted(set(files))[:limit]
if not files:
    raise SystemExit("no calibration images found")
with open(os.path.join(output, "list.txt"), "w") as listing:
    for index, filename in enumerate(files):
        image = Image.open(filename).convert("RGB")
        ratio = min(w / image.width, h / image.height)
        nw, nh = round(image.width * ratio), round(image.height * ratio)
        canvas = Image.new("RGB", (w, h), (114, 114, 114))
        canvas.paste(image.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
        array = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        raw = os.path.join(output, f"{index}.raw")
        array.tofile(raw)
        listing.write(raw + "\n")
print(len(files), "calibration images")
PY
    QUANT_ARGS=(--input_dlc "$DLC" --output_dlc "$WORK/${NAME}_int8.dlc"
                --input_list "$CALIB_WORK/list.txt" --act_bitwidth 8 --weights_bitwidth 8)
    if [ "$FLOAT_IO" = 1 ]; then
        QUANT_ARGS+=(--preserve_io_datatype)
    else
        QUANT_ARGS+=(--preserve_io_datatype qnn_boxes qnn_scores)
    fi
    say "qairt-quantizer (INT8)"
    "$QUANTIZER" "${QUANT_ARGS[@]}"
    DLC="$WORK/${NAME}_int8.dlc"
fi

"$TOOLS/qairt-dlc-info" -i "$DLC" > "$WORK/$NAME.dlc-info.txt" 2>&1 || true
if [ "$PRECISION" = int8 ]; then
    grep -q "act_bitwidth=8" "$WORK/$NAME.dlc-info.txt" \
        || die "INT8 verification failed: act_bitwidth=8 not found"
    grep -q "weights_bitwidth=8" "$WORK/$NAME.dlc-info.txt" \
        || die "INT8 verification failed: weights_bitwidth=8 not found"
    grep -Eq "data type: [us]Fxp_8" "$WORK/$NAME.dlc-info.txt" \
        || die "INT8 verification failed: no 8-bit fixed-point tensors found"
    say "verified: INT8 weights and activations are present in the DLC"
fi
GRAPH="$(grep -oiE 'graph[ _]?name[^A-Za-z0-9_]+[A-Za-z0-9_.:-]+' "$WORK/$NAME.dlc-info.txt" \
    | head -1 | grep -oE '[A-Za-z0-9_.:-]+$' || true)"

python3 - "$WORK/htp_ext.json" "$WORK/ctx_config.json" "$ARCH" "$GRAPH" "$LIB" <<'PY'
import json, os, sys
ext_path, config_path, arch, graph, lib = sys.argv[1:]
ext = {"devices": [{"dsp_arch": arch}]}
if graph:
    ext["graphs"] = [{"graph_names": [graph], "O": 3}]
with open(ext_path, "w") as f:
    json.dump(ext, f, indent=2)
config = {"backend_extensions": {
    "shared_library_path": os.path.join(lib, "libQnnHtpNetRunExtensions.so"),
    "config_file_path": os.path.abspath(ext_path),
}}
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
PY

mkdir -p "$WORK/out"
say "preparing HTP $ARCH context binary"
"$GENERATOR" \
    --backend "$LIB/libQnnHtp.so" \
    --model "$LIB/libQnnModelDlc.so" \
    --dlc_path "$DLC" \
    --binary_file "$NAME" \
    --output_dir "$WORK/out" \
    --config_file "$WORK/ctx_config.json"

test -s "$WORK/out/$NAME.bin" || die "context generator did not produce $NAME.bin"
cp "$WORK/out/$NAME.bin" "$OUT/$NAME.bin"
say "wrote $OUT/$NAME.bin ($(du -h "$OUT/$NAME.bin" | cut -f1))"
printf '\nUse in models.json: "model": "%s.bin", "backend": "qnn"\n' "$NAME"
