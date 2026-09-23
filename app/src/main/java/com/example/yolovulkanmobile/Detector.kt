package com.example.yolovulkanmobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF

interface Detector : AutoCloseable {
    val backendLabel: String
    fun detect(bitmap: Bitmap): List<Detection>
}

class LoadResult(val detector: Detector?, val error: String?)

internal fun detectionsFromFlat(flat: FloatArray, labels: List<String>): List<Detection> {
    val out = ArrayList<Detection>(flat.size / 6)
    var i = 0
    while (i + 5 < flat.size) {
        val label = flat[i + 4].toInt()
        out.add(
            Detection(
                rect = RectF(flat[i], flat[i + 1], flat[i] + flat[i + 2], flat[i + 1] + flat[i + 3]),
                label = label,
                score = flat[i + 5],
                labelName = labels.getOrElse(label) { "id$label" },
            )
        )
        i += 6
    }
    return out
}

class NcnnDetector(override val backendLabel: String) : Detector {

    override fun detect(bitmap: Bitmap): List<Detection> = YoloNcnn.detect(bitmap)

    override fun close() = YoloNcnn.release()
}

object DetectorFactory {
    fun create(context: Context, spec: ModelSpec): LoadResult = when (spec.backend) {
        "ncnn" ->
            if (YoloNcnn.init(context.assets, spec, useGpu = true)) {
                val runtime = when {
                    YoloNcnn.useGpu -> "GPU · Vulkan"
                    YoloNcnn.hasGpu() -> "CPU (GPU available)"
                    else -> "CPU (no Vulkan GPU)"
                }
                LoadResult(NcnnDetector("$runtime · ${YoloNcnn.nativePrecision()}"), null)
            } else {
                LoadResult(null, "missing ${spec.paramAsset} / ${spec.binAsset}?")
            }
        "qnn" -> QnnDetector.create(context, spec)
        else -> LoadResult(null, "unknown backend '${spec.backend}'")
    }
}
