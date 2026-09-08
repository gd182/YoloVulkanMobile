# assets/

`models.json` и `coco.txt` включены в репозиторий. **Веса моделей (`*.param` /
`*.bin`) не включаются** — они перечислены в `.gitignore`, чтобы исходный
код оставался под Apache-2.0. Перед сборкой поместите файлы, на которые
ссылаются записи в `models.json`, в эту папку.

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
