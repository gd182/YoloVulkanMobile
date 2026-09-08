package com.example.yolovulkanmobile

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import org.json.JSONObject

data class Detection(
    val rect: RectF,
    val label: Int,
    val score: Float,
    val labelName: String,
)

data class ModelSpec(
    val id: String,
    val displayName: String,
    val paramAsset: String,
    val binAsset: String,
    val labels: List<String>,
    val inputName: String,
    val outputName: String,
    val targetSize: Int,
    val decoded: Boolean,
    val bgr: Boolean,
    val confThreshold: Float,
    val nmsThreshold: Float,
) {
    val numClass: Int get() = labels.size
}

object YoloNcnn {

    private const val CONFIG_ASSET = "models.json"

    init {
        System.loadLibrary("yolovulkan")
    }

    @Volatile
    var useGpu: Boolean = false
        private set

    @Volatile
    var currentSpec: ModelSpec? = null
        private set

    fun loadConfig(assets: AssetManager): Pair<List<ModelSpec>, String> {
        val root = JSONObject(assets.open(CONFIG_ASSET).bufferedReader().use { it.readText() })
        val arr = root.getJSONArray("models")
        val specs = (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            ModelSpec(
                id = o.getString("id"),
                displayName = o.optString("displayName", o.getString("id")),
                paramAsset = o.getString("param"),
                binAsset = o.getString("bin"),
                labels = readLabels(assets, o.get("labels")),
                inputName = o.optString("inputName", "images"),
                outputName = o.optString("outputName", "output"),
                targetSize = o.optInt("targetSize", 640),
                decoded = o.optBoolean("decoded", false),
                bgr = o.optBoolean("bgr", true),
                confThreshold = o.optDouble("confThreshold", 0.25).toFloat(),
                nmsThreshold = o.optDouble("nmsThreshold", 0.45).toFloat(),
            )
        }
        val default = root.optString("default").ifEmpty { specs.first().id }
        return specs to default
    }

    private fun readLabels(assets: AssetManager, node: Any): List<String> = when (node) {
        is org.json.JSONArray -> (0 until node.length()).map { node.getString(it) }
        is String -> assets.open(node).bufferedReader().use { r ->
            r.readLines().map { it.trim() }.filter { it.isNotEmpty() }
        }
        else -> error("models.json: 'labels' must be an array or an asset filename")
    }

    external fun nativeHasGpu(): Boolean

    private external fun nativeInit(
        assetManager: AssetManager,
        paramAsset: String,
        binAsset: String,
        inputName: String,
        outputName: String,
        targetSize: Int,
        numClass: Int,
        decoded: Boolean,
        bgr: Boolean,
        useGpu: Boolean,
    ): Boolean

    private external fun nativeDetect(
        bitmap: Bitmap,
        probThreshold: Float,
        nmsThreshold: Float,
    ): FloatArray

    fun hasGpu(): Boolean = nativeHasGpu()

    @Synchronized
    fun init(assets: AssetManager, spec: ModelSpec, useGpu: Boolean = true): Boolean {
        this.useGpu = useGpu && hasGpu()
        val ok = nativeInit(
            assets, spec.paramAsset, spec.binAsset, spec.inputName, spec.outputName,
            spec.targetSize, spec.numClass, spec.decoded, spec.bgr, this.useGpu,
        )
        currentSpec = if (ok) spec else null
        return ok
    }

    @Synchronized
    fun detect(bitmap: Bitmap): List<Detection> {
        val spec = currentSpec ?: return emptyList()
        val flat = nativeDetect(bitmap, spec.confThreshold, spec.nmsThreshold)
        val out = ArrayList<Detection>(flat.size / 6)
        var i = 0
        while (i + 5 < flat.size) {
            val label = flat[i + 4].toInt()
            out.add(
                Detection(
                    rect = RectF(flat[i], flat[i + 1], flat[i] + flat[i + 2], flat[i + 1] + flat[i + 3]),
                    label = label,
                    score = flat[i + 5],
                    labelName = spec.labels.getOrElse(label) { "id$label" },
                )
            )
            i += 6
        }
        return out
    }
}
