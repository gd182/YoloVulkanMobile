<h1 align="center">VulkanSight</h1>

<p align="center">Детекция объектов YOLO в реальном времени на Android с использованием Vulkan-бэкенда ncnn.</p>

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

<p align="center"><a href="README.md">English</a> · <b>Русский</b></p>

## О проекте

VulkanSight выполняет детекцию объектов YOLO на превью камеры в реальном времени.
Инференс предпочитает GPU через Vulkan-бэкенд **ncnn**, с автоматическим
откатом на CPU при отсутствии совместимого GPU.

Модели задаются в одном `models.json` в ассетах приложения - добавить или
переключить модель можно без правок кода (см. раздел «Модели»). Поддерживаются
оба популярных варианта экспорта для ncnn: сырой DFL-head (один `output`) и
нативный экспорт Ultralytics (`format=ncnn`).

## Как это работает

```
CameraX ImageAnalysis (RGBA_8888)
  → вертикальный ARGB_8888 Bitmap              MainActivity.kt
  → JNI                                        YoloNcnn.kt
  → ncnn::Net  (opt.use_vulkan_compute = true) cpp/yolo.cpp
       letterbox → forward → decode → NMS
  → float[x, y, w, h, label, score] per box
  → OverlayView рисует рамки поверх превью      OverlayView.kt
```

- **Нативная часть** (`app/src/main/cpp/`) — `yolo.cpp` реализует детектор без
  OpenCV; `yolo_jni.cpp` — JNI-мост, который управляет инстансом Vulkan.
- **ncnn** — в репозитории есть prebuilt-релиз `android-vulkan` `20260526`,
  распакованный по ABI в `app/src/main/cpp/ncnn/<abi>/`; CMake находит его с
  помощью `find_package(ncnn)`.
- **Бэкенд** — выбирается Vulkan, если `ncnn::get_gpu_count() > 0`, иначе — CPU.
  Статус-строка показывает выбранный бэкенд, FPS и задержку инференса.

## Модели

Все модели описаны в `app/src/main/assets/models.json`. Выпадающий список внизу
экрана выбирает активную модель; поле `default` задаёт модель по умолчанию.

Поля включают `param`, `bin`, `inputName` / `outputName`, `targetSize`,
`decoded`, `bgr`, `confThreshold`, `nmsThreshold` и `labels`.

Ключевые различия:

| Поле | `decoded: false` | `decoded: true` |
|---|---|---|
| Источник | сырой DFL-head (single-`output`) | Ultralytics `format=ncnn` |
| `inputName` / `outputName` | `images` / `output` | `in0` / `out0` |
| Декод рамок | выполняется на устройстве | уже включён в граф |
| Паддинг | до кратного 32 | до полного квадрата `targetSize` |
| `bgr` | `true` | `false` |

### Добавление модели

Экспортируйте модель в ncnn, скопируйте `.param` и `.bin` в
`app/src/main/assets/` и добавьте соответствующую запись в `models.json`.
Убедитесь, что `targetSize` совпадает с `imgsz` при экспорте.

Если рамки смещены или неправильного масштаба, сначала проверьте соответствие
`targetSize` и `imgsz` — это самая частая причина.

## Сборка и запуск

Требуется Android Studio (AGP 9.4.0) и NDK. `minSdk 24`, `compileSdk 37`.

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Собираемые ABI: `arm64-v8a`, `armeabi-v7a`, `x86_64`.

Prebuilt ncnn хранится в репозитории; веса моделей в репозиторий не попадают (как их добавить, см. `app/src/main/assets/README.md`). Приложение
собирается и запускается без весов; при отсутствии файла модель покажет
"load failed".

## Структура проекта

```
app/src/main/
  java/com/example/yolovulkanmobile/
  cpp/
  assets/
  res/
```

## Лицензия

Apache-2.0 — см. [LICENSE](LICENSE) и [NOTICE](NOTICE).

Prebuilt ncnn — BSD-3-Clause. Примеры весов `yolov8*` сконвертированы из
Ultralytics YOLO и распространяются под AGPL-3.0; перед распространением
замените их на модели с более мягкой лицензией, если это необходимо.
adaptive-слои не нужны. Для страницы в Google Play дополнительно нужен PNG
**512 × 512** (в APK не входит).

Название приложения на экране — `app_name` в
`app/src/main/res/values/strings.xml`.

## Структура проекта

```
app/src/main/
  java/com/example/yolovulkanmobile/
    MainActivity.kt        настройка CameraX, кадр → bitmap, список моделей, статус-строка
    YoloNcnn.kt            обёртка JNI; парсит models.json в ModelSpec
    OverlayView.kt         рисует рамки поверх превью (маппинг center-crop)
  cpp/
    CMakeLists.txt         find_package(ncnn), собирает libyolovulkan.so
    yolo.h / yolo.cpp      детектор без OpenCV: letterbox, DFL + decoded головы, NMS
    yolo_jni.cpp           JNI-мост; владеет инстансом Vulkan
    ncnn/<abi>/            prebuilt ncnn android-vulkan 20260526
  assets/
    models.json           реестр моделей
    coco.txt              имена 80 классов COCO
    *.param / *.bin       веса моделей
  res/layout/activity_main.xml   PreviewView + OverlayView + Spinner
```

## Лицензия

Apache-2.0 — см. [LICENSE](LICENSE) и [NOTICE](NOTICE).

Prebuilt ncnn — под BSD-3-Clause. Веса `yolov8*` сконвертированы из Ultralytics
YOLO и распространяются под **AGPL-3.0** — они лежат в репозитории только как
рабочий пример; перед распространением приложения замени модель на
permissive-лицензированную (NanoDet, YOLOX, RT-DETR).
