package com.yolov8.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.yolov8.demo.camera.CameraManager
import com.yolov8.demo.detector.YOLOv8Detector
import com.yolov8.demo.utils.FPSMonitor
import com.yolov8.demo.view.DetectionOverlayView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var detectionOverlay: DetectionOverlayView
    private lateinit var fpsText: TextView
    private lateinit var inferenceText: TextView
    private lateinit var modeText: TextView
    private lateinit var switchModeBtn: Button

    private lateinit var cameraManager: CameraManager
    private var detector: YOLOv8Detector? = null
    private val fpsMonitor = FPSMonitor()

    private var useNNAPI = false
    private var isProcessing = false
    private var lastBitmap: android.graphics.Bitmap? = null

    private val activityResultLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val cameraGranted = permissions[Manifest.permission.CAMERA] ?: false
            if (cameraGranted) {
                startDetection()
            } else {
                Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_SHORT).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        detectionOverlay = findViewById(R.id.detectionOverlay)
        fpsText = findViewById(R.id.fpsText)
        inferenceText = findViewById(R.id.inferenceText)
        modeText = findViewById(R.id.modeText)
        switchModeBtn = findViewById(R.id.switchModeBtn)

        switchModeBtn.setOnClickListener {
            toggleMode()
        }

        checkPermissions()
    }

    private fun checkPermissions() {
        val requiredPermissions = mutableListOf(Manifest.permission.CAMERA)

        val missingPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            activityResultLauncher.launch(missingPermissions.toTypedArray())
        } else {
            startDetection()
        }
    }

    private fun startDetection() {
        lifecycleScope.launch {
            initDetector()
            initCamera()
            updateModeText()
        }
    }

    private suspend fun initDetector() = withContext(Dispatchers.IO) {
        detector?.close()
        detector = YOLOv8Detector(this@MainActivity, useNNAPI)
        detector?.init()
    }

    private fun initCamera() {
        cameraManager = CameraManager(this, this, previewView)
        cameraManager.setOnFrameListener { bitmap ->
            if (!isProcessing) {
                isProcessing = true
                lastBitmap = bitmap
                processFrame(bitmap)
            }
        }
        cameraManager.startCamera()
    }

    private fun processFrame(bitmap: android.graphics.Bitmap) {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val result = detector?.detect(bitmap)

                withContext(Dispatchers.Main) {
                    result?.let {
                        detectionOverlay.setResults(it.results, bitmap.width, bitmap.height)
                        fpsMonitor.addInferenceTime(it.inferenceTimeMs.toLong())
                    }

                    fpsMonitor.update()

                    fpsText.text = getString(R.string.fps_format, fpsMonitor.getFPS())
                    inferenceText.text = getString(R.string.inference_time_format, fpsMonitor.getAvgInferenceTime())
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                isProcessing = false
            }
        }
    }

    private fun toggleMode() {
        useNNAPI = !useNNAPI
        isProcessing = false
        fpsMonitor.reset()

        lifecycleScope.launch {
            initDetector()
            updateModeText()
            Toast.makeText(this@MainActivity,
                if (useNNAPI) R.string.nnapi_mode else R.string.cpu_mode,
                Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateModeText() {
        modeText.text = if (useNNAPI) getString(R.string.nnapi_mode) else getString(R.string.cpu_mode)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        detector?.close()
    }
}
