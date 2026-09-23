package com.example.yolovulkanmobile

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Build
import android.util.Log

object QnnNative {
    init {
        System.loadLibrary("yolovulkan")
    }

    external fun nativeInit(
        assetManager: AssetManager,
        modelAsset: String,
        nativeLibDir: String,
        numClass: Int,
        boxesNormalized: Boolean,
    ): String

    external fun nativeIsBuilt(): Boolean

    external fun nativeDescribe(): String

    external fun nativeRelease()

    external fun nativeDetect(bitmap: Bitmap, probThreshold: Float, nmsThreshold: Float): FloatArray
}

class QnnDetector private constructor(
    private val spec: ModelSpec,
    override val backendLabel: String,
) : Detector {

    override fun detect(bitmap: Bitmap): List<Detection> =
        detectionsFromFlat(
            QnnNative.nativeDetect(bitmap, spec.confThreshold, spec.nmsThreshold),
            spec.labels,
        )

    override fun close() = QnnNative.nativeRelease()

    companion object {
        fun isSupported(context: Context, spec: ModelSpec): Boolean {
            val arm64 = Build.SUPPORTED_ABIS.any { it == "arm64-v8a" }
            val socManufacturer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                Build.SOC_MANUFACTURER
            } else {
                ""
            }
            val hardware = listOf(Build.HARDWARE, Build.BOARD, Build.DEVICE)
                .joinToString(" ")
                .lowercase()
            val qualcomm = socManufacturer.contains("qualcomm", ignoreCase = true) ||
                listOf("qcom", "qualcomm", "kona", "lahaina", "taro", "kalama", "pineapple")
                    .any { it in hardware }
            val modelPresent = runCatching {
                context.assets.open(spec.modelAsset).use { }
            }.isSuccess
            val nativeBuilt = arm64 && runCatching { QnnNative.nativeIsBuilt() }.getOrDefault(false)
            val supported = nativeBuilt && qualcomm && modelPresent

            if (!supported) {
                Log.i(
                    "QnnDetector",
                    "QNN hidden: arm64=$arm64 nativeBuilt=$nativeBuilt " +
                        "qualcomm=$qualcomm modelPresent=$modelPresent " +
                        "soc=$socManufacturer hardware=$hardware",
                )
            }
            return supported
        }

        fun create(context: Context, spec: ModelSpec): LoadResult {
            val error = QnnNative.nativeInit(
                context.assets,
                spec.modelAsset,
                context.applicationInfo.nativeLibraryDir,
                spec.numClass,
                spec.boxesNormalized,
            )
            if (error.isNotEmpty()) return LoadResult(null, error)
            val runtime = QnnNative.nativeDescribe()
            val actualPrecision = when {
                Regex("""\bin \[[^]]+] (?:u8q|i8q)\b""").containsMatchIn(runtime) -> "INT8"
                Regex("""\bin \[[^]]+] (?:u16q|i16q)\b""").containsMatchIn(runtime) -> "INT16"
                Regex("""\bin \[[^]]+] fp16\b""").containsMatchIn(runtime) -> "FP16"
                Regex("""\bin \[[^]]+] fp32\b""").containsMatchIn(runtime) -> "FP32"
                else -> "unknown precision"
            }
            Log.i(
                "QnnDetector",
                "model=${spec.id} actualPrecision=$actualPrecision runtime={$runtime}",
            )
            return LoadResult(
                QnnDetector(spec, "NPU · Qualcomm HTP (QNN) · $actualPrecision"),
                null,
            )
        }
    }
}
