package com.example.yolovulkanmobile

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.yolovulkanmobile.databinding.ActivityMainBinding
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var lastFpsTs = 0L
    private var frames = 0

    private lateinit var specs: List<ModelSpec>
    @Volatile private var specIndex = 0
    private val switching = AtomicBoolean(false)
    private var spinnerInitialCallback = true

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val (loaded, defaultId) = YoloNcnn.loadConfig(assets)
        specs = loaded
        specIndex = loaded.indexOfFirst { it.id == defaultId }.coerceAtLeast(0)

        setupModelSpinner()
        loadSpec(specIndex)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupModelSpinner() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            specs.map { it.displayName },
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }

        binding.modelSpinner.adapter = adapter
        binding.modelSpinner.setSelection(specIndex, false)
        binding.modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (spinnerInitialCallback) {
                    spinnerInitialCallback = false
                    return
                }
                if (position != specIndex) loadSpec(position)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun loadSpec(index: Int) {
        if (!switching.compareAndSet(false, true)) {
            binding.modelSpinner.setSelection(specIndex, false)
            return
        }
        specIndex = index
        val spec = specs[index]
        binding.statusText.text = "loading ${spec.displayName}…"
        analysisExecutor.execute {
            val ok = YoloNcnn.init(assets, spec, useGpu = true)
            switching.set(false)
            runOnUiThread {
                binding.overlay.setResults(emptyList(), 1, 1)
                binding.statusText.text = if (ok) {
                    "${spec.displayName}\n${backendLabel()}"
                } else {
                    "load failed: ${spec.displayName}\nmissing ${spec.paramAsset} / ${spec.binAsset}?"
                }
            }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = binding.previewView.surfaceProvider
            }

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        android.util.Size(1280, 720),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                    )
                )
                .build()

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::analyze) }

            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        if (switching.get() || !busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        try {
            val bitmap = image.toUprightBitmap()
            val started = System.nanoTime()
            val detections = YoloNcnn.detect(bitmap)
            val ms = (System.nanoTime() - started) / 1_000_000.0

            runOnUiThread {
                binding.overlay.setResults(detections, bitmap.width, bitmap.height)
                updateFps(ms)
            }
        } catch (t: Throwable) {
            Log.e("MainActivity", "analyze failed", t)
        } finally {
            busy.set(false)
            image.close()
        }
    }

    private fun backendLabel(): String = when {
        YoloNcnn.useGpu -> "GPU · Vulkan"
        YoloNcnn.hasGpu() -> "CPU (GPU available)"
        else -> "CPU (no Vulkan GPU)"
    }

    private fun updateFps(inferMs: Double) {
        frames++
        val now = System.currentTimeMillis()
        if (lastFpsTs == 0L) lastFpsTs = now
        if (now - lastFpsTs >= 1000) {
            val fps = frames * 1000f / (now - lastFpsTs)
            val name = YoloNcnn.currentSpec?.displayName ?: "model"
            binding.statusText.text =
                "%s\n%s · %.1f FPS · %.0f ms".format(name, backendLabel(), fps, inferMs)
            Log.i("MainActivity", "%s | %s | %.1f FPS | %.0f ms".format(name, backendLabel(), fps, inferMs))
            frames = 0
            lastFpsTs = now
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }
}

private fun ImageProxy.toUprightBitmap(): Bitmap {
    val plane = planes[0]
    val bmp = Bitmap.createBitmap(
        plane.rowStride / plane.pixelStride,
        height,
        Bitmap.Config.ARGB_8888,
    )
    bmp.copyPixelsFromBuffer(plane.buffer)
    val cropped = if (bmp.width != width) {
        Bitmap.createBitmap(bmp, 0, 0, width, height)
    } else {
        bmp
    }

    val rotation = imageInfo.rotationDegrees
    if (rotation == 0) return cropped
    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    return Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
}
