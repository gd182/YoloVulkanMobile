# assets/

`models.example.json` и `labels.txt` включены в репозиторий. Рабочий `models.json`
локальный и находится в `.gitignore`, поэтому ваши модели и названия классов не
попадут в Git. Если `models.json` отсутствует, приложение использует
`models.example.json`. **Веса моделей (`*.param` /
`*.bin`) не включаются** — они перечислены в `.gitignore`, чтобы исходный
код оставался под Apache-2.0. Перед сборкой поместите файлы, на которые
ссылаются записи в `models.json`, в эту папку.

Для собственной конфигурации начните с копии примера:

```bash
cp app/src/main/assets/models.example.json app/src/main/assets/models.json
```

Приложение собирается и запускается без весов: если файл модели отсутствует,
в статус-строке появится "load failed" и боксы не будут отрисованы.

## Как добавить модель

Экспортируйте модель в ncnn, затем скопируйте `.param` и `.bin` в
`app/src/main/assets/`:

```bash
yolo export model=best.pt format=ncnn imgsz=640 half=True
cp best_ncnn_model/model.ncnn.param best_ncnn_model/model.ncnn.bin app/src/main/assets/
```

Добавьте запись в `models.json` с соответствующими полями:

- `"decoded": true`, `"bgr": false`, `"inputName": "in0"`, `"outputName": "out0"`
- `"targetSize"` — должен совпадать с `imgsz` при экспорте
- `labels` — список классов в том же порядке, что и в `best_ncnn_model/metadata.yaml`

Подробное описание полей — в корневом `README.md`. Экспорт Ultralytics YOLO
распространяется под **AGPL-3.0** — подходит для локального тестирования, но не для распространения (см. `NOTICE`).

Разрядность в конфигурации не указывается: приложение определяет её при загрузке
из ncnn `.bin` или из описания входного QNN-тензора.

---

## QNN (NPU) context binaries / Контекст-бинари QNN

A model with `"backend": "qnn"` uses a single pre-compiled QNN context binary
(`"model": "model_v81.bin"`) instead of `.param` + `.bin`. It is git-ignored like the other
weights. How to build one: `docs/QNN.md`.

Модель с `"backend": "qnn"` использует один заранее скомпилированный QNN context binary
(`"model": "model_v81.bin"`) вместо пары `.param` + `.bin`. Он в `.gitignore`, как и остальные
веса. Как собрать: `docs/QNN.ru.md`.
