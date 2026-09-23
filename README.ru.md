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
Для инференса используется **ncnn** через Vulkan/CPU или опциональный нативный
бэкенд **Qualcomm QNN** для NPU Hexagon на поддерживаемых устройствах Snapdragon.

Модели задаются в одном `models.json` в ассетах приложения - добавить или
переключить модель можно без правок кода (см. раздел «Модели»). Поддерживаются
оба популярных варианта экспорта для ncnn: сырой DFL-head (один `output`) и
нативный экспорт Ultralytics (`format=ncnn`).

## Как это работает

```
CameraX ImageAnalysis (RGBA_8888)
  → вертикальный ARGB_8888 Bitmap              MainActivity.kt
  → Detector (ncnn или QNN)                    Detector.kt
  → JNI                                        YoloNcnn.kt / QnnDetector.kt
  → ncnn::Net или QNN graphExecute             cpp/yolo.cpp / cpp/qnn_yolo.cpp
       letterbox → inference → decode → NMS
  → float[x, y, w, h, label, score] per box
  → OverlayView рисует рамки поверх превью      OverlayView.kt
```

- **Нативная часть** (`app/src/main/cpp/`) — `yolo.cpp` реализует детектор без
  OpenCV; `yolo_jni.cpp` — JNI-мост, который управляет инстансом Vulkan.
- **ncnn** — в репозитории есть prebuilt-релиз `android-vulkan` `20260526`,
  распакованный по ABI в `app/src/main/cpp/ncnn/<abi>/`; CMake находит его с
  помощью `find_package(ncnn)`.
- **Бэкенд** — выбирается Vulkan, если `ncnn::get_gpu_count() > 0`, иначе — CPU.
  QNN-модели показываются только при доступности QNN. Статус-строка показывает
  выбранный бэкенд, автоматически определённую разрядность модели, FPS и задержку инференса.

## Модели

Все модели описаны в локальном `app/src/main/assets/models.json`. Он находится
в `.gitignore`; для начала скопируйте `models.example.json`. Если локального
файла нет, приложение читает отслеживаемый пример. Выпадающий список внизу
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

## NPU (Qualcomm QNN)

На устройствах Snapdragon модель может выполняться на NPU Hexagon через нативный QNN C API
(`"backend": "qnn"` в `models.json`) с заранее скомпилированным context binary.
Настройка и конвертация: [docs/QNN.ru.md](docs/QNN.ru.md).

## Сборка и запуск

Требуется Android Studio (AGP 9.4.0) и NDK. `minSdk 24`, `compileSdk 37`.

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Собираемые ABI: `arm64-v8a`, `armeabi-v7a`, `x86_64`. QNN доступен только в
`arm64-v8a` и только если `qnn.sdk.dir` или `QNN_SDK_ROOT` указывает на
совпадающую версию QAIRT SDK; в остальных сборках продолжают работать ncnn-бэкенды.

Prebuilt ncnn хранится в репозитории; веса моделей в репозиторий не попадают (как их добавить, см. `app/src/main/assets/README.md`). Приложение
собирается и запускается без весов; при отсутствии файла модель покажет
"load failed".

## Структура проекта

```
app/src/main/
  java/com/example/yolovulkanmobile/
    MainActivity.kt        настройка CameraX, кадр → bitmap, список моделей, статус-строка
    Detector.kt            общий интерфейс детектора
    YoloNcnn.kt            ncnn/Vulkan-реализация и разбор models.json
    QnnDetector.kt         QNN-реализация и проверка доступности NPU
    OverlayView.kt         рисует рамки поверх превью (маппинг center-crop)
  cpp/
    CMakeLists.txt         find_package(ncnn), собирает libyolovulkan.so
    yolo.h / yolo.cpp      детектор без OpenCV: letterbox, DFL + decoded головы, NMS
    yolo_jni.cpp           JNI-мост; владеет инстансом Vulkan
    qnn_yolo.* / qnn_jni.cpp  QNN context binary, graphExecute и JNI-мост
    ncnn/<abi>/            prebuilt ncnn android-vulkan 20260526
  assets/
    models.json            реестр моделей
    labels.txt             имена 80 классов COCO
    *.param / *.bin        локальные веса моделей (не коммитятся)
  res/layout/activity_main.xml   PreviewView + OverlayView + Spinner
docs/                      инструкции по QNN на русском и английском
scripts/                   конвертация моделей в QNN context binary
```

## Лицензия

Apache-2.0 — см. [LICENSE](LICENSE) и [NOTICE](NOTICE).

Prebuilt ncnn распространяется под BSD-3-Clause. Весов моделей в репозитории
нет. Перед распространением приложения проверьте лицензию добавленных весов.
Компоненты Qualcomm QNN остаются под условиями поставляемой с ними лицензии.
