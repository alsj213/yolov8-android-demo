package com.yolov8.demo.detector

import android.content.Context
import android.graphics.Bitmap
import com.yolov8.demo.R
import com.taobao.android.mnn.MNNForwardType
import com.taobao.android.mnn.MNNNetInstance
import com.yolov8.demo.utils.ImageUtils
import com.yolov8.demo.utils.NMSUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * MNN (Mobile Neural Network) 推理框架检测器
 * 由阿里开源，性能优异，支持多种硬件加速
 */
class MNNDetector(
    private val context: Context,
    private val numThreads: Int = 4,
    private val useVulkan: Boolean = false,  // 使用 Vulkan GPU 加速
    private val useOpenCL: Boolean = false  // 使用 OpenCL GPU 加速
) : Detector {

    private var mnnNet: MNNNetInstance? = null
    private var mnnSession: MNNNetInstance.Session? = null
    private var inputTensor: MNNNetInstance.Session.Tensor? = null
    private var outputTensor: MNNNetInstance.Session.Tensor? = null

    override val inputSize = 640
    val confThreshold = 0.4f
    val iouThreshold = 0.45f

    override val backendName = buildString {
        append("MNN")
        if (useVulkan) append("-Vulkan")
        if (useOpenCL) append("-OpenCL")
        if (!useVulkan && !useOpenCL) append("-CPU")
        append(" ($numThreads-threads)")
    }

    // COCO 80 classes
    private val classNames = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    )

    override suspend fun init(): Unit = withContext(Dispatchers.IO) {
        try {
            android.util.Log.e("MNN", "========== 开始初始化 MNN 检测器 ==========")
            android.util.Log.e("MNN", "后端名称: $backendName")

            // 先加载 MNN 库
            android.util.Log.e("MNN", "1. 开始加载 MNN 库...")
            if (!com.taobao.android.mnn.MNNNetNative.loadLibraries()) {
                throw RuntimeException("Failed to load MNN libraries")
            }
            android.util.Log.e("MNN", "   MNN 库加载成功 ✓")

            // 复制模型到 files 目录
            android.util.Log.e("MNN", "2. 开始加载模型文件...")
            val modelFile = getModelFile()
            android.util.Log.e("MNN", "   模型路径: ${modelFile.absolutePath}")
            android.util.Log.e("MNN", "   模型大小: ${modelFile.length()} 字节")

            // 加载 MNN 模型
            android.util.Log.e("MNN", "3. 开始创建 MNNNetInstance...")
            mnnNet = MNNNetInstance.createFromFile(modelFile.absolutePath)
            if (mnnNet == null) {
                throw RuntimeException("Failed to load MNN model: MNNNetInstance.createFromFile returned null")
            }
            android.util.Log.e("MNN", "   MNNNetInstance 创建成功 ✓")

            // 创建 session
            android.util.Log.e("MNN", "4. 开始创建 Session...")
            val config = MNNNetInstance.Config()
            config.numThread = numThreads
            val forwardType = when {
                useVulkan -> MNNForwardType.FORWARD_VULKAN.type
                useOpenCL -> MNNForwardType.FORWARD_OPENCL.type
                else -> MNNForwardType.FORWARD_CPU.type
            }
            config.forwardType = forwardType
            android.util.Log.e("MNN", "   线程数: $numThreads, forwardType: $forwardType")

            mnnSession = mnnNet!!.createSession(config)
            if (mnnSession == null) {
                throw RuntimeException("Failed to create MNN session: createSession returned null")
            }
            android.util.Log.e("MNN", "   Session 创建成功 ✓")

            // 获取输入输出张量
            android.util.Log.e("MNN", "5. 获取输入输出张量...")
            inputTensor = mnnSession!!.getInput("images")
            android.util.Log.e("MNN", "   输入张量: ${if (inputTensor != null) "✓" else "✗ (null)"}")
            android.util.Log.e("MNN", "   输入张量维度: ${inputTensor?.getDimensions()?.contentToString()}")

            outputTensor = mnnSession!!.getOutput("output0")
            if (outputTensor == null) {
                android.util.Log.w("MNN", "   output0 不存在，尝试 output...")
                outputTensor = mnnSession!!.getOutput("output")
            }
            android.util.Log.e("MNN", "   输出张量: ${if (outputTensor != null) "✓" else "✗ (null)"}")
            android.util.Log.e("MNN", "   输出张量维度: ${outputTensor?.getDimensions()?.contentToString()}")

            android.util.Log.e("MNN", "========== MNN 初始化成功 ==========")

        } catch (e: UnsatisfiedLinkError) {
            android.util.Log.e("MNN", "MNN 库加载失败 (UnsatisfiedLinkError): ${e.message}", e)
            throw RuntimeException("MNN library not available", e)
        } catch (e: Exception) {
            android.util.Log.e("MNN", "MNN 初始化异常: ${e.message}", e)
            e.printStackTrace()
            throw RuntimeException("Failed to initialize MNN: ${e.message}", e)
        }
    }

    private fun getModelFile(): File {
        val modelFileName = "yolov8n_mnn.mnn"
        val outputFile = File(context.filesDir, modelFileName)

        if (!outputFile.exists()) {
            android.util.Log.d("MNN", "正在复制 MNN 模型文件...")
            try {
                // 先尝试从 raw 资源加载 MNN 模型
                val resourceId = R.raw.yolov8n_mnn
                if (resourceId != 0) {
                    context.resources.openRawResource(resourceId).use { input ->
                        FileOutputStream(outputFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                } else {
                    // 如果没有 MNN 模型，抛出异常让上层处理
                    throw RuntimeException("MNN model not found. Please convert YOLOv8n ONNX to MNN format.")
                }
            } catch (e: Exception) {
                android.util.Log.e("MNN", "复制模型失败: ${e.message}")
                throw e
            }
        }
        return outputFile
    }

    override suspend fun detect(bitmap: Bitmap): DetectionResult = withContext(Dispatchers.Default) {
        val session = mnnSession ?: return@withContext DetectionResult(emptyList(), 0f, 1f, 0f, 0f)
        val input = inputTensor ?: return@withContext DetectionResult(emptyList(), 0f, 1f, 0f, 0f)
        val output = outputTensor ?: return@withContext DetectionResult(emptyList(), 0f, 1f, 0f, 0f)

        val startTime = System.nanoTime()

        // Letterbox 预处理
        val preprocessResult = ImageUtils.bitmapToFloatBuffer(bitmap, inputSize)
        val floatArray = preprocessResult.floatArray

        // 设置输入数据
        input.setInputFloatData(floatArray)

        // 执行推理
        session.run()

        // 获取输出数据
        val outputData = output.floatData

        val inferenceTime = (System.nanoTime() - startTime) / 1_000_000f

        // 后处理 - 解析 YOLOv8 输出
        val detections = postProcess(outputData, preprocessResult.scale, preprocessResult.padX, preprocessResult.padY)

        DetectionResult(
            detections = detections,
            inferenceTimeMs = inferenceTime,
            scale = preprocessResult.scale,
            padX = preprocessResult.padX,
            padY = preprocessResult.padY
        )
    }

    private fun postProcess(outputData: FloatArray, scale: Float, padX: Float, padY: Float): List<DetectorResult> {
        val results = mutableListOf<DetectorResult>()

        // YOLOv8 输出格式: [1, 84, 8400] - 展平后是一维数组
        // 84 = 4 bbox (cx, cy, w, h) + 80 class confidences
        // 8400 = 检测框数量
        val numClasses = 80
        val numBoxes = 8400

        for (i in 0 until numBoxes) {
            val offset = i

            // 提取 bbox (cx, cy, w, h)
            val cx = outputData[offset]
            val cy = outputData[offset + numBoxes]
            val w = outputData[offset + numBoxes * 2]
            val h = outputData[offset + numBoxes * 3]

            // 找到最大类别置信度
            var maxConf = 0f
            var maxClassId = 0
            for (c in 0 until numClasses) {
                val conf = outputData[offset + numBoxes * (4 + c)]
                if (conf > maxConf) {
                    maxConf = conf
                    maxClassId = c
                }
            }

            if (maxConf >= confThreshold) {
                // 转换为 x1, y1, x2, y2 (相对于模型输入尺寸)
                val x1 = (cx - w / 2)
                val y1 = (cy - h / 2)
                val x2 = (cx + w / 2)
                val y2 = (cy + h / 2)

                // 转换回原始图像尺寸 (移除 letterbox)
                val origX1 = (x1 - padX) / scale
                val origY1 = (y1 - padY) / scale
                val origX2 = (x2 - padX) / scale
                val origY2 = (y2 - padY) / scale

                results.add(
                    DetectorResult(
                        classId = maxClassId,
                        className = classNames[maxClassId],
                        confidence = maxConf,
                        x1 = origX1,
                        y1 = origY1,
                        x2 = origX2,
                        y2 = origY2
                    )
                )
            }
        }

        // NMS 过滤
        return NMSUtils.nonMaxSuppression(results, iouThreshold)
    }

    override fun close() {
        mnnSession?.release()
        mnnSession = null
        mnnNet?.release()
        mnnNet = null
        inputTensor = null
        outputTensor = null
    }

    companion object {
        // 检查 MNN 库是否可用
        val isMnnAvailable: Boolean by lazy {
            com.taobao.android.mnn.MNNNetNative.loadLibraries()
        }
    }
}
