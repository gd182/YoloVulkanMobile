package com.example.yolovulkanmobile

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
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
    private var warmupFrames = 0
    private var cameraBitmap: Bitmap? = null

    private lateinit var specs: List<ModelSpec>
    @Volatile private var specIndex = 0
    private val switching = AtomicBoolean(false)
    private var spinnerUserAction = false
    @Volatile private var detector: Detector? = null
    @Volatile private var currentSpec: ModelSpec? = null
    @Volatile private var loadError: String? = null

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
        specs = loaded.filter { spec ->
            spec.backend != "qnn" || QnnDetector.isSupported(applicationContext, spec)
        }
        check(specs.isNotEmpty()) { "models.json has no models supported on this device" }
        specIndex = specs.indexOfFirst { it.id == defaultId }.coerceAtLeast(0)

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
        binding.modelSpinner.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) spinnerUserAction = true
            false
        }
        binding.modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!spinnerUserAction) return
                spinnerUserAction = false
                if (position != specIndex) loadSpec(position)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                spinnerUserAction = false
            }
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
            val loadStarted = System.nanoTime()
            detector?.close()
            detector = null
            currentSpec = null
            val result = DetectorFactory.create(applicationContext, spec)
            detector = result.detector
            currentSpec = if (result.detector != null) spec else null
            loadError = result.error?.let { "load failed: ${spec.displayName}\n$it" }
            if (result.detector == null) Log.e("MainActivity", "load failed: ${spec.displayName}: ${result.error}")
            val loadMs = (System.nanoTime() - loadStarted) / 1_000_000.0
            runOnUiThread {
                frames = 0
                lastFpsTs = 0L
                warmupFrames = if (result.detector != null) 5 else 0
                binding.overlay.setResults(emptyList(), 1, 1)
                binding.statusText.text = if (result.detector != null) {
                    "%s\n%s · loaded %.0f ms · warming up".format(
                        spec.displayName, result.detector.backendLabel, loadMs
                    )
                } else {
                    "load failed: ${spec.displayName}\n${result.error}"
                }
                Log.i("MainActivity", "${spec.displayName} loaded in %.0f ms".format(loadMs))
                switching.set(false)
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
                .setOutputImageRotationEnabled(true)
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
            val bitmap = image.copyToBitmap()
            val started = System.nanoTime()
            val detections = detector?.detect(bitmap) ?: emptyList()
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

    private fun updateFps(inferMs: Double) {
        if (detector == null) {
            loadError?.let { binding.statusText.text = it }
            return
        }
        if (warmupFrames > 0) {
            warmupFrames--
            if (warmupFrames == 0) {
                frames = 0
                lastFpsTs = System.currentTimeMillis()
            }
            return
        }
        frames++
        val now = System.currentTimeMillis()
        if (lastFpsTs == 0L) lastFpsTs = now
        if (now - lastFpsTs >= 1000) {
            val fps = frames * 1000f / (now - lastFpsTs)
            val name = currentSpec?.displayName ?: "model"
            val backend = detector?.backendLabel ?: "-"
            binding.statusText.text =
                "%s\n%s · %.1f FPS · %.0f ms".format(name, backend, fps, inferMs)
            Log.i("MainActivity", "%s | %s | %.1f FPS | %.0f ms".format(name, backend, fps, inferMs))
            frames = 0
            lastFpsTs = now
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.execute {
            detector?.close()
            detector = null
        }
        analysisExecutor.shutdown()
        cameraBitmap = null
    }

    private fun ImageProxy.copyToBitmap(): Bitmap {
        val plane = planes[0]
        val bufferWidth = plane.rowStride / plane.pixelStride
        val reusable = cameraBitmap?.takeIf {
            it.width == bufferWidth && it.height == height
        } ?: Bitmap.createBitmap(bufferWidth, height, Bitmap.Config.ARGB_8888).also {
            cameraBitmap = it
        }

        plane.buffer.rewind()
        reusable.copyPixelsFromBuffer(plane.buffer)
        return if (bufferWidth == width) {
            reusable
        } else {
            Bitmap.createBitmap(reusable, 0, 0, width, height)
        }
    }
}
