<h1 align="center">VulkanSight</h1>

<p align="center">Real-time YOLO object detection on Android using ncnn's Vulkan backend.</p>

###

<div align="center">
  <img src="https://skillicons.dev/icons?i=kotlin" height="40" alt="Kotlin logo" />
  <img width="12" />
  <img src="https://skillicons.dev/icons?i=cpp" height="40" alt="C++ logo" />
  <img width="12" />
  <img src="https://skillicons.dev/icons?i=cmake" height="40" alt="CMake logo" />
  <img width="12" />
  <img src="https://skillicons.dev/icons?i=gradle" height="40" alt="Gradle logo" />
  <img width="12" />
  <img src="https://skillicons.dev/icons?i=androidstudio" height="40" alt="Android Studio logo" />
</div>

###

<p align="center"><b>English</b> · <a href="README.ru.md">Русский</a></p>

## Overview

VulkanSight runs YOLO object detection on a phone's camera preview. Inference
executes on the GPU via **ncnn**'s Vulkan compute backend, with an automatic
CPU fallback if no compatible GPU is available.

Models are configurable from a single `models.json` in the app assets — no
code changes are required to add or switch a model (see the Models section).
Both common ncnn export styles are supported: raw DFL head (single `output`) and
Ultralytics' native `format=ncnn` export.

## Architecture

```
CameraX ImageAnalysis (RGBA_8888)
  → upright ARGB_8888 Bitmap                    MainActivity.kt
  → JNI                                         YoloNcnn.kt
  → ncnn::Net  (opt.use_vulkan_compute = true)  cpp/yolo.cpp
       letterbox → forward → decode → NMS
  → float[x, y, w, h, label, score] per box
  → OverlayView draws boxes over the preview    OverlayView.kt
```

- **Native** (`app/src/main/cpp/`) — `yolo.cpp` implements the detector
  without OpenCV; `yolo_jni.cpp` is the JNI bridge and manages the Vulkan
  instance (`JNI_OnLoad` / `JNI_OnUnload`).
- **ncnn** — prebuilt `android-vulkan` release `20260526` is tracked per ABI
  under `app/src/main/cpp/ncnn/<abi>/`; CMake finds it via `find_package(ncnn)`.
- **Backend selection** — uses Vulkan if `ncnn::get_gpu_count() > 0`, otherwise
  falls back to CPU. The UI status line shows backend, FPS and inference latency.

## Models

Models are declared in `app/src/main/assets/models.json`. The bottom dropdown
selects the active model; `default` sets which model loads at startup.

```jsonc
{
  "default": "coco-n",
  "models": [ ... ]
}
```

Example fields include `param`, `bin`, `inputName` / `outputName`, `targetSize`,
`decoded`, `bgr`, `confThreshold`, `nmsThreshold` and `labels`.

Key differences:

| Field | `decoded: false` | `decoded: true` |
|---|---|---|
| Source | raw DFL head (single-`output`) | Ultralytics `format=ncnn` |
| `inputName` / `outputName` | `images` / `output` | `in0` / `out0` |
| Box decode | performed on-device | baked into the graph |
| Padding | padded to multiple of 32 | padded to full `targetSize` square |
| `bgr` | `true` | `false` |

### Adding your own model

Export an ncnn model, copy the `.param` / `.bin` files into `app/src/main/assets/`
and add an entry to `models.json` with matching fields. Ensure `targetSize`
matches the export `imgsz`.

If boxes are shifted or scaled incorrectly, check that `targetSize` equals the
export `imgsz` — this is the most common cause.

## Build & run

Requires Android Studio (AGP 9.4.0) and the NDK. `minSdk 24`, `compileSdk 37`.

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Built ABIs: `arm64-v8a`, `armeabi-v7a`, `x86_64`.

Model weights are intentionally not committed (see
`app/src/main/assets/README.md`) - the app builds without them; a missing
model file results in a "load failed" status.

## Project layout

```
app/src/main/
  java/com/example/yolovulkanmobile/
  cpp/
  assets/
  res/
```

See in-source files for details.

## License

Apache-2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The included ncnn prebuilt is BSD-3-Clause. Example `yolov8*` weights are
converted from Ultralytics and are AGPL-3.0; replace them with permissively
licensed models before redistributing the app.
