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
import com.yolov8.demo.detector.Detector
import com.yolov8.demo.detector.MNNDetector
import com.yolov8.demo.detector.YOLOv8Detector
import com.yolov8.demo.utils.FPSMonitor
import com.yolov8.demo.utils.ImageUtils
import com.yolov8.demo.view.DetectionOverlayView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    val TAG = "YOLOv8"

    private lateinit var previewView: PreviewView
    private lateinit var detectionOverlay: DetectionOverlayView
    private lateinit var fpsText: TextView
    private lateinit var inferenceText: TextView
    private lateinit var modeText: TextView
    private lateinit var switchModeBtn: Button

    private lateinit var cameraManager: CameraManager
    private var detector: Detector? = null
    private val fpsMonitor = FPSMonitor()

    // 推理后端类型
    private enum class BackendType {
        ORT_CPU,        // ONNX Runtime CPU
        ORT_NNAPI,      // ONNX Runtime NNAPI
        ORT_QNN,        // ONNX Runtime QNN
        MNN_CPU,        // MNN CPU
        MNN_VULKAN,     // MNN Vulkan GPU
        MNN_OPENCL      // MNN OpenCL GPU
    }

    // 后端列表 - 顺序决定切换顺序
    private val backends = listOf(
        BackendType.ORT_CPU,
        BackendType.MNN_CPU,
        BackendType.ORT_NNAPI,
        BackendType.ORT_QNN,
        // BackendType.MNN_VULKAN,
        // BackendType.MNN_OPENCL
    )

    private var currentBackendIndex = 0

    @Volatile
    private var isProcessing = false
    @Volatile
    private var isSwitchingMode = false
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
        android.util.Log.d("YOLOv8", "========= onCreate 开始 =========")
        android.util.Log.d("YOLOv8", "后端列表: $backends, 数量: ${backends.size}")
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        detectionOverlay = findViewById(R.id.detectionOverlay)
        fpsText = findViewById(R.id.fpsText)
        inferenceText = findViewById(R.id.inferenceText)
        modeText = findViewById(R.id.modeText)
        switchModeBtn = findViewById(R.id.switchModeBtn)

        // 先更新UI，确保Activity显示正常
        fpsText.text = "FPS: --"
        inferenceText.text = "推理: --ms"

        Toast.makeText(this, "请确保已授予相机权限", Toast.LENGTH_LONG).show()

        switchModeBtn.setOnClickListener {
            toggleMode()
        }

        checkPermissions()
    }

    private fun checkPermissions() {
        android.util.Log.d("YOLOv8", "检查权限...")
        android.util.Log.d("YOLOv8", "后端列表: $backends, 数量: ${backends.size}")
        val requiredPermissions = mutableListOf(Manifest.permission.CAMERA)

        val missingPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        android.util.Log.d("YOLOv8", "缺失权限: $missingPermissions")

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
        val backend = backends[currentBackendIndex]
        android.util.Log.d("YOLOv8", "开始初始化检测器, 后端=$backend")
        try {
            detector?.close()

            // 根据后端类型创建检测器
            detector = when (backend) {
                BackendType.ORT_CPU -> {
                    YOLOv8Detector(this@MainActivity, useNNAPI = false, useQNN = false)
                }
                BackendType.ORT_NNAPI -> {
                    YOLOv8Detector(this@MainActivity, useNNAPI = true, useQNN = false)
                }
                BackendType.ORT_QNN -> {
                    YOLOv8Detector(this@MainActivity, useNNAPI = false, useQNN = true)
                }
                BackendType.MNN_CPU -> {
                    MNNDetector(this@MainActivity, useVulkan = false, useOpenCL = false)
                }
                BackendType.MNN_VULKAN -> {
                    MNNDetector(this@MainActivity, useVulkan = true, useOpenCL = false)
                }
                BackendType.MNN_OPENCL -> {
                    MNNDetector(this@MainActivity, useVulkan = false, useOpenCL = true)
                }
            }

            detector?.init()
            android.util.Log.d("YOLOv8", "检测器初始化成功: ${detector?.backendName}")

        } catch (e: Exception) {
            android.util.Log.e("YOLOv8", "检测器初始化失败: ${e.message}", e)

            // 如果当前后端失败，回退到 ORT_CPU
            if (backend != BackendType.ORT_CPU) {
                android.util.Log.w("YOLOv8", "$backend 不可用，回退到 ORT_CPU")
                currentBackendIndex = 0
                detector = YOLOv8Detector(this@MainActivity, useNNAPI = false, useQNN = false)
                detector?.init()
            }
        }
        return@withContext
    }

    private fun initCamera() {
        cameraManager = CameraManager(this, this, previewView)
        cameraManager.setOnFrameListener { bitmap ->
            if (!isProcessing && !isSwitchingMode) {
                isProcessing = true
                lastBitmap = bitmap
                processFrame(bitmap)
            }
        }
        cameraManager.startCamera()
    }

    private var firstFrame = true

    private fun processFrame(bitmap: android.graphics.Bitmap) {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                // 检查 detector 是否可用
                val currentDetector = detector ?: return@launch

                val result = currentDetector.detect(bitmap)

                // 打印调试信息
                if (firstFrame) {
                    firstFrame = false
                    android.util.Log.d("YOLOv8", "===== 推理结果 =====")
                    android.util.Log.d("YOLOv8", "推理时间: ${result.inferenceTimeMs}ms")
                    android.util.Log.d("YOLOv8", "检测到物体数量: ${result.detections.size}")
                }

                withContext(Dispatchers.Main) {
                    detectionOverlay.setResults(result.detections, bitmap.width, bitmap.height)
                    fpsMonitor.addInferenceTime(result.inferenceTimeMs.toLong())
                    fpsMonitor.update()

                    fpsText.text = getString(R.string.fps_format, fpsMonitor.getFPS())
                    inferenceText.text = getString(R.string.inference_time_format, fpsMonitor.getAvgInferenceTime())
                }
            } catch (e: Exception) {
                android.util.Log.e("YOLOv8", "推理错误", e)
                e.printStackTrace()
            } finally {
                isProcessing = false
            }
        }
    }

    private fun toggleMode() {
        if (isSwitchingMode) return
        android.util.Log.e("YOLOv8", "========== 切换模式, 当前索引: $currentBackendIndex, 后端总数: ${backends.size}")
        android.util.Log.e("YOLOv8", "后端列表: $backends")

        isSwitchingMode = true
        fpsMonitor.reset()

        lifecycleScope.launch {
            // 等待当前帧处理完成
            while (isProcessing) {
                kotlinx.coroutines.delay(10)
            }

            // 切换到下一个后端
            currentBackendIndex = (currentBackendIndex + 1) % backends.size
            android.util.Log.e("YOLOv8", "切换到索引: $currentBackendIndex, 后端: ${backends[currentBackendIndex]}")
            initDetector()

            withContext(Dispatchers.Main) {
                updateModeText()
                val modeName = getBackendDisplayName(backends[currentBackendIndex])
                Toast.makeText(this@MainActivity, "$modeName (共${backends.size}个后端)", Toast.LENGTH_SHORT).show()
                isSwitchingMode = false
            }
        }
    }

    private fun getBackendDisplayName(backend: BackendType): String {
        return when (backend) {
            BackendType.ORT_CPU -> "ORT-CPU"
            BackendType.ORT_NNAPI -> "ORT-NNAPI"
            BackendType.ORT_QNN -> "ORT-QNN"
            BackendType.MNN_CPU -> "MNN-CPU"
            BackendType.MNN_VULKAN -> "MNN-Vulkan"
            BackendType.MNN_OPENCL -> "MNN-OpenCL"
            else -> "Unknown"
        }
    }

    private fun updateModeText() {
        val backend = backends[currentBackendIndex]
        modeText.text = getBackendDisplayName(backend)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        detector?.close()
    }
}
