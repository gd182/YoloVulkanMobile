# Бэкенд Qualcomm NPU (QNN)

🇬🇧 [English version](QNN.md)

Весь граф YOLO выполняется на NPU Hexagon (HTP) через **нативный QNN C API**
(`app/src/main/cpp/qnn_yolo.cpp`): заранее скомпилированный *context binary* грузится один
раз, дальше каждый кадр — один `graphExecute`. Без прослойки LiteRT/TFLite, без CPU-частей
графа, без компиляции на устройстве.

- Пре- и постпроцессинг (letterbox, деквантизация, декод, NMS) — на C++.
- Размер входа, layout (NHWC/NCHW), тип данных и квантизация читаются из бинаря.
- Поддерживаемые типы ввода/вывода: FP32, FP16, INT8/UINT8, INT16/UINT16 (scale/offset).
- Выход может быть одним декодированным тензором Ultralytics `[1, 4+nc, N]` /
  `[1, N, 4+nc]` либо раздельными `boxes` `[1,4,N]` и `scores` `[1,nc,N]`:
  `xywh` в пикселях входа плюс оценки классов. Экспорт без NMS.
- Только arm64. На других устройствах/ABI QNN-модель скрыта или покажет «load failed», а
  модели на Vulkan/ncnn продолжат работать.

## Из чего это состоит

| Компонент | Откуда |
|---|---|
| Runtime-библиотеки (`libQnnHtp`, `libQnnHtpV*Stub/Skel`, `libQnnSystem`) | Maven `com.qualcomm.qti:qnn-runtime`, версия `qnn` в `gradle/libs.versions.toml` |
| Заголовки (только для сборки) | QAIRT SDK, путь в `local.properties` |
| Инструменты конвертации | QAIRT SDK (запускает `scripts/build_qnn_model.sh`) |
| Context binary | по одному на архитектуру HTP, собирает скрипт |

**Версия SDK, которым собран бинарь, должна совпадать с версией runtime** (`qnn` в
`libs.versions.toml`, сейчас `2.50.0`). Если у вас SDK другой версии — поменяйте `qnn` на
такую же, иначе `contextCreateFromBinary` упадёт.

## 1. SDK и Gradle

Установите QAIRT SDK:
<https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk> (нужен
Qualcomm ID и принятие лицензии). Затем в `local.properties` (в `.gitignore`):

```
qnn.sdk.dir=/путь/к/Qualcomm/AIStack/QAIRT/2.50.0
```

или `export QNN_SDK_ROOT=...` / `-Pqnn.sdk.dir=...`. В логе CMake появится `QNN: enabled`.
Без этого приложение собирается как обычно, а QNN-модели скрыты.

## 2. Собрать context binary — одной командой

На Ubuntu x86_64 сборка выполняется нативно, без Docker:

```bash
sudo apt install python3 python3-venv python3-pip build-essential libgl1 libglib2.0-0 libtinfo6
export QNN_SDK_ROOT="$HOME/Qualcomm/AIStack/QAIRT/2.50.0"
scripts/build_qnn_model_ubuntu_x86.sh \
  --model build/qnn/best8_v81/in/best.pt \
  --arch v81 --imgsz 640 --precision int8 \
  --calib calibration --calib-count 300 \
  --name best_model_android_htp
```

Если SDK распакован в другое место, передайте его явно:

```bash
scripts/build_qnn_model_ubuntu_x86.sh --sdk /полный/путь/к/QAIRT/2.50.0 ...
```

Корень SDK — это каталог, внутри которого находится `include/QNN/QnnInterface.h`.

Скрипт использует x86_64-инструменты из QAIRT SDK и сразу создаёт HTP context binary.
Версия SDK должна совпадать с `qnn-runtime` приложения.

```bash
scripts/build_qnn_model.sh --model best.pt --imgsz 640
```

Что делает скрипт (всё нужное докачивается при первом запуске):

1. Находит SDK и архитектуру HTP подключённого телефона (`ro.soc.model`).
2. На macOS ставит через Homebrew **Colima + Docker CLI** (спросит, либо `-y`) и запускает
   нативную arm64-VM — Rosetta не нужна.
3. В контейнере Ubuntu 24.04 arm64: экспортирует ONNX (если вход `.pt`), конвертирует в DLC
   через `qairt-converter` (FP16) или квантизует через `qairt-quantizer` (INT8).
4. По `adb` кладёт DLC и Android-инструменты SDK на телефон, запускает там
   `qnn-context-binary-generator` (точный SoC, без эмуляции) и забирает результат.
5. Кладёт `<имя>_<arch>.bin` в `app/src/main/assets/` и печатает запись для `models.json`.

Опции: `--precision int8 --calib DIR` (калибровочные картинки, см. ниже), `--arch v79`,
`--name`, `--out`, `--float-io`, `--prepare device|x86`, `--shell`. Подробнее:
`scripts/build_qnn_model.sh --help`.

Телефон должен быть подключён (`adb devices`). Без него скрипт переходит на
`--prepare x86`: x86_64-инструмент SDK в эмулируемом контейнере; на Apple Silicon для этого
нужна Rosetta (`softwareupdate --install-rosetta`, затем Colima с `--vz-rosetta`) — обычный
QEMU падает с `hogl::ring: failed to init ring mutex`.

| SoC | Архитектура HTP |
|---|---|
| Snapdragon 888 | v68 |
| 8 Gen 1 | v69 |
| 8 Gen 2 | v73 |
| 8 Gen 3 | v75 |
| 8 Elite | v79 |
| 8 Elite Gen 5 (Adreno 840) | v81 |

### FP16 или INT8

- **FP16** не требует данных и стоит по умолчанию.
- **INT8** быстрее. Передайте в `--calib` папку со 100–300 репрезентативными jpg/png (все
  классы, расстояния, освещение); скрипт делает letterbox так же, как приложение, и отдаёт
  их `qairt-quantizer`. После этого проверьте точность (`yolo val` на ONNX как FP32-эталон
  против нескольких кадров на телефоне).

### Те же шаги вручную

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

Конвертер работает на Linux (x86_64 или arm64-библиотеки, которые SDK поставляет для
Python 3.12); генератор должен выполняться либо на телефоне (`bin/aarch64-android`, плюс
`libQnnHtp`, `libQnnHtpPrepare`, `libQnnModelDlc`, `libQnnSystem`,
`libQnnHtpNetRunExtensions`, `libQnnHtpV*Stub` и
`lib/hexagon-v*/unsigned/libQnnHtpV*Skel.so`, с `ADSP_LIBRARY_PATH` на эту папку), либо на
x86_64 Linux (`bin/x86_64-linux-clang`). Сборка генератора для aarch64 Ubuntu на обычном
ARM-хосте не работает (`No Snapdragon SOC detected`).

## 3. Подключить к приложению

`scripts/build_qnn_model.sh` уже скопировал бинарь в `app/src/main/assets/` (в
`.gitignore`). Поправьте запись `qnn` в `assets/models.json`:

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

В `labels` должно быть ровно столько имён, сколько классов у модели.
`boxesNormalized: true` — только если граф выдаёт координаты 0..1.

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat -s QnnYolo MainActivity
```

В статус-строке будет `NPU · Qualcomm HTP (QNN)`, в logcat —
`ready: in [1,640,640,3] u8q | out [1,9,8400] fp16`.

## Если что-то не работает

| Сообщение | Причина |
|---|---|
| модели нет в выпадающем списке | не arm64-устройство Qualcomm, бинаря нет в assets, либо приложение собрано без SDK (причину пишет `QnnDetector.isSupported` в logcat, тег `QnnDetector`) |
| `built without the QNN SDK` | не задан `qnn.sdk.dir` или указана не та папка |
| `dlopen libQnnHtp.so failed` | устройство не arm64 |
| `contextCreateFromBinary failed: 30001` | неверная конфигурация HTP cache либо бинарь собран под другую HTP-архитектуру/версию QAIRT; пересоберите совпадающим скриптом и runtime |
| `no compatible QNN backend API version` | заголовки SDK старее runtime-библиотек |
| `output dims [...] do not match 4+N classes` | число `labels` не совпадает с моделью, либо в графе NMS / другая голова |
| `unsupported ... tensor type` | тип ввода/вывода вне списка выше |
| скрипт: `module 'onnx' has no attribute 'version'` | не та версия onnx в окружении конвертера; скрипт фиксирует `onnx==1.16.1` (из `sdk.yaml` SDK) |
| скрипт: `hogl::ring: failed to init ring mutex` | x86_64-инструмент под QEMU; используйте `--prepare device` или Rosetta |

## Что ещё не сделано (следующие шаги по скорости)

- Режим питания HTP (DCVS/burst) из приложения — сейчас частоты определяет системный governor.
- Zero-copy ввод/вывод через общие буферы `rpcmem`.
- Конвейер: препроцессинг кадра N+1 параллельно с выполнением кадра N на NPU.
- Препроцессинг — главная нагрузка на CPU (замер на Snapdragon 8 Elite Gen 5, 416 px, FP16:
  препроцессинг ~8 мс, NPU ~8 мс, постпроцессинг ~1,3 мс; logcat, тег `QnnYolo`, печатает
  средние значения каждые 90 кадров). Следующий шаг: resize и letterbox в uint8 сразу во входной тензор.

## Лицензия

Библиотеки QNN runtime — проприетарные (Qualcomm AI Stack License). Они подтягиваются из
Maven при сборке и **не** коммитятся в этот репозиторий. Условия распространения задаёт
лицензия внутри AAR `qnn-runtime`; прочитайте её перед публикацией APK.
