# Qualcomm NPU (QNN) backend

🇷🇺 [Русская версия](QNN.ru.md)

Runs the whole YOLO graph on the Hexagon NPU (HTP) through the **native QNN C API**
(`app/src/main/cpp/qnn_yolo.cpp`): a pre-compiled *context binary* is loaded once,
then every frame is one `graphExecute`. No LiteRT/TFLite layer, no CPU partitions,
no on-device graph compilation.

- Pre/post-processing (letterbox, dequantize, decode, NMS) is C++.
- Input size, layout (NHWC/NCHW), dtype and quantization are read from the binary.
- Supported I/O dtypes: FP32, FP16, INT8/UINT8, INT16/UINT16 (scale/offset).
- Output may be one Ultralytics-decoded tensor `[1, 4+nc, N]` / `[1, N, 4+nc]`,
  or separate boxes `[1,4,N]` and scores `[1,nc,N]`: `xywh` in input pixels plus
  class scores. Export without NMS.
- Arm64 only. On other devices/ABIs the QNN model is hidden or shows "load failed" and the
  Vulkan/ncnn models keep working.

## Pieces and versions

| Piece | Where it comes from |
|---|---|
| Runtime libs (`libQnnHtp`, `libQnnHtpV*Stub/Skel`, `libQnnSystem`) | Maven `com.qualcomm.qti:qnn-runtime`, version `qnn` in `gradle/libs.versions.toml` |
| Headers (build time only) | QAIRT SDK, path in `local.properties` |
| Model tools | QAIRT SDK (run by `scripts/build_qnn_model.sh`) |
| Context binary | one per HTP arch, built by the script |

**The SDK version used to build the binary must equal the runtime version** (`qnn` in
`libs.versions.toml`, currently `2.50.0`). If your SDK is another version, change `qnn` to
match — otherwise `contextCreateFromBinary` fails.

## 1. SDK and Gradle

Install the QAIRT SDK from
<https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk> (Qualcomm ID +
license acceptance). Then in `local.properties` (git-ignored):

```
qnn.sdk.dir=/path/to/Qualcomm/AIStack/QAIRT/2.50.0
```

or `export QNN_SDK_ROOT=...` / `-Pqnn.sdk.dir=...`. The CMake log prints `QNN: enabled`.
Without it the app builds normally and QNN models are hidden.

## 2. Build the context binary — one command

```bash
scripts/build_qnn_model.sh --model best.pt --imgsz 640
```

What it does (everything is downloaded on the first run):

1. Finds the SDK and the HTP arch of the connected phone (`ro.soc.model`).
2. On macOS installs **Colima + Docker CLI** through Homebrew (asks first, or `-y`) and
   starts a native arm64 VM — no Rosetta needed.
3. In an Ubuntu 24.04 arm64 container: exports ONNX (`.pt` input), converts it to a DLC
   with `qairt-converter` (FP16), or quantizes with `qairt-quantizer` (INT8).
4. Copies the DLC and the SDK's Android tools to the phone over `adb`, runs
   `qnn-context-binary-generator` there (exact SoC, no emulation) and pulls the result.
5. Puts `<name>_<arch>.bin` into `app/src/main/assets/` and prints the `models.json` entry.

Options: `--precision int8 --calib DIR` (calibration images, see below), `--arch v79`,
`--name`, `--out`, `--float-io`, `--prepare device|x86`, `--shell`. See
`scripts/build_qnn_model.sh --help`.

The phone must be connected (`adb devices`). Without it the script falls back to
`--prepare x86`, which runs the x86_64 SDK tool in an emulated container; on Apple Silicon
that needs Rosetta (`softwareupdate --install-rosetta`, then Colima with `--vz-rosetta`) —
plain QEMU aborts with `hogl::ring: failed to init ring mutex`.

| SoC | HTP arch |
|---|---|
| Snapdragon 888 | v68 |
| 8 Gen 1 | v69 |
| 8 Gen 2 | v73 |
| 8 Gen 3 | v75 |
| 8 Elite | v79 |
| 8 Elite Gen 5 (Adreno 840) | v81 |

### FP16 vs INT8

- **FP16** needs no data and is the default.
- **INT8** is faster. Point `--calib` at a folder with 100–300 representative jpg/png images
  (all classes, distances, lighting); the script letterboxes them exactly like the app and
  feeds them to `qairt-quantizer`. Check accuracy afterwards (`yolo val` on the ONNX as the
  FP32 baseline vs a few frames on the phone).

### The same steps by hand

```bash
qairt-converter --input_network best.onnx --output_path best.dlc \
    --source_model_input_shape images 1,3,640,640 --float_bitwidth 16 --float_bias_bitwidth 16
# INT8: qairt-quantizer --input_dlc best.dlc --output_dlc best_int8.dlc \
#           --input_list calib.txt --act_bitwidth 8 --weights_bitwidth 8

cat > htp_ext.json <<'EOF'
{ "devices": [{ "dsp_arch": "v81" }] }
EOF
cat > ctx_config.json <<'EOF'
{ "backend_extensions": { "shared_library_path": "libQnnHtpNetRunExtensions.so",
                          "config_file_path": "htp_ext.json" } }
EOF

qnn-context-binary-generator --backend libQnnHtp.so --model libQnnModelDlc.so \
    --dlc_path best.dlc --binary_file model_v81 --output_dir out --config_file ctx_config.json
```

The converter runs on Linux (x86_64, or the arm64 libraries the SDK ships for Python 3.12);
the generator must run either on the phone (`bin/aarch64-android`, plus `libQnnHtp`,
`libQnnHtpPrepare`, `libQnnModelDlc`, `libQnnSystem`, `libQnnHtpNetRunExtensions`,
`libQnnHtpV*Stub` and `lib/hexagon-v*/unsigned/libQnnHtpV*Skel.so`, with `ADSP_LIBRARY_PATH`
set to that folder) or on x86_64 Linux (`bin/x86_64-linux-clang`). The aarch64 Ubuntu build
of the generator does not work on a generic ARM host (`No Snapdragon SOC detected`).

## 3. Add to the app

`scripts/build_qnn_model.sh` already copied the binary into `app/src/main/assets/`
(git-ignored). Edit the `qnn` entry in `assets/models.json`:

```json
{
  "id": "qnn",
  "displayName": "YOLO11 · NPU (QNN)",
  "backend": "qnn",
  "model": "best_v81.bin",
  "boxesNormalized": false,
  "confThreshold": 0.3,
  "nmsThreshold": 0.45,
  "labels": ["class0", "class1"]
}
```

`labels` must have exactly as many entries as the model has classes.
`boxesNormalized: true` only if your graph outputs 0..1 coordinates.

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat -s QnnYolo MainActivity
```

The status line shows `NPU · Qualcomm HTP (QNN)`; logcat prints
`ready: in [1,640,640,3] u8q | out [1,9,8400] fp16`.

## Troubleshooting

| Message | Cause |
|---|---|
| model missing from the dropdown | not an arm64 Qualcomm device, the binary is not in assets, or the app was built without the SDK (`QnnDetector.isSupported` logs the reason under tag `QnnDetector`) |
| `built without the QNN SDK` | `qnn.sdk.dir` not set, or wrong folder |
| `dlopen libQnnHtp.so failed` | not an arm64 device |
| `contextCreateFromBinary failed: 30001` | invalid HTP cache configuration, or a binary built for another HTP arch / QAIRT version; rebuild with the matching script and runtime |
| `no compatible QNN backend API version` | SDK headers are older than the runtime libs |
| `output dims [...] do not match 4+N classes` | `labels` count differs from the model, or the graph includes NMS / another head |
| `unsupported ... tensor type` | I/O dtype outside the list above |
| script: `module 'onnx' has no attribute 'version'` | wrong onnx in the converter env; the script pins `onnx==1.16.1` (from the SDK's `sdk.yaml`) |
| script: `hogl::ring: failed to init ring mutex` | x86_64 tool under QEMU; use `--prepare device` or Rosetta |

## Not done yet (next performance steps)

- HTP DCVS/burst power config from the app (clocks currently follow the system governor).
- Zero-copy I/O through `rpcmem` shared buffers.
- Pipelining preprocessing of frame N+1 with NPU execution of frame N.
- Preprocessing is the biggest CPU cost (measured on Snapdragon 8 Elite Gen 5, 416 px, FP16:
  preprocess ~8 ms, NPU execute ~8 ms, postprocess ~1.3 ms; logcat tag `QnnYolo` prints the
  averages every 90 frames). Next step: resize and letterbox in uint8 straight into the input tensor.

## Licensing

The QNN runtime libraries are proprietary (Qualcomm AI Stack License). They are fetched from
Maven at build time and are **not** committed to this repository. Redistribution is governed
by the license bundled with the `qnn-runtime` AAR; read it before publishing an APK.
