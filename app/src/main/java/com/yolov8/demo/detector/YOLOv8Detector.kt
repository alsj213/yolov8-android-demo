package com.yolov8.demo.detector

import android.content.Context
import android.os.Build
import android.os.SystemClock
import ai.onnxruntime.*
import com.yolov8.demo.utils.ImageUtils
import com.yolov8.demo.utils.NMSUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

class YOLOv8Detector(
    private val context: Context,
    private val useNNAPI: Boolean = false,
    private val useQNN: Boolean = false,
    private val numThreads: Int = 4
) : Detector {
    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null

    override val inputSize = 640
    val confThreshold = 0.4f  // 提高置信度阈值，减少误检
    val iouThreshold = 0.45f

    override val backendName = buildString {
        append("ORT-")
        when {
            useQNN -> append("QNN")
            useNNAPI -> append("NNAPI")
            else -> append("CPU")
        }
        if (!useNNAPI && !useQNN) {
            append("($numThreads-threads)")
        }
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

    override suspend fun init() = withContext(Dispatchers.IO) {
        try {
            close()

            if (ortEnvironment == null) {
                ortEnvironment = OrtEnvironment.getEnvironment()
            }

            val sessionOptions = OrtSession.SessionOptions()

            when {
                useQNN -> {
                    sessionOptions.setIntraOpNumThreads(1)
                    sessionOptions.setInterOpNumThreads(1)
                    sessionOptions.addNnapi()
                }
                useNNAPI -> {
                    sessionOptions.setIntraOpNumThreads(1)
                    sessionOptions.setInterOpNumThreads(1)
                    sessionOptions.addNnapi()
                }
                else -> {
                    sessionOptions.setIntraOpNumThreads(numThreads)
                    sessionOptions.setInterOpNumThreads(1)
                }
            }

            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            val modelFile = getModelFile()
            ortSession = ortEnvironment!!.createSession(modelFile.absolutePath, sessionOptions)

        } catch (e: Exception) {
            throw RuntimeException("Failed to initialize ONNX Runtime: ${e.message}", e)
        }
    }

    private fun getModelFile(): File {
        val modelFileName = "yolov8n.onnx"
        val outputFile = File(context.filesDir, modelFileName)

        if (!outputFile.exists()) {
            val resourceId = context.resources.getIdentifier("yolov8n", "raw", context.packageName)
            context.resources.openRawResource(resourceId).use { input ->
                FileOutputStream(outputFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outputFile
    }

    override suspend fun detect(bitmap: android.graphics.Bitmap): DetectionResult = withContext(Dispatchers.Default) {
        val startTime = SystemClock.elapsedRealtimeNanos()

        // Letterbox 预处理，保持宽高比
        val preprocessResult = ImageUtils.bitmapToFloatBuffer(bitmap, inputSize)
        val floatArray = preprocessResult.floatArray

        val inputName = ortSession!!.inputNames.first()
        val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())

        // 创建 FloatBuffer
        val buffer = java.nio.ByteBuffer.allocateDirect(floatArray.size * 4)
            .order(java.nio.ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(floatArray)
            .rewind() as java.nio.FloatBuffer

        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment,
            buffer,
            inputShape
        )

        val output = ortSession!!.run(mapOf(inputName to inputTensor))
        // Shape: [1, 84, 8400] - batch, features, anchors
        val rawOutput = output.get(0).value as Array<*>  // 第一层: batch dimension
        val outputTensor = rawOutput[0] as Array<FloatArray>  // 第二层: [84, 8400]

        // 坐标从640x640空间映射回原始图像空间
        // 先减去padding，再除以scale
        val results = parseOutput(outputTensor, preprocessResult.scale, preprocessResult.padX, preprocessResult.padY)

        val inferenceTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000f

        output.close()
        inputTensor.close()

        DetectionResult(results, inferenceTime, preprocessResult.scale, preprocessResult.padX, preprocessResult.padY)
    }

    private fun parseOutput(
        output: Array<FloatArray>,
        scale: Float,
        padX: Float,
        padY: Float
    ): List<DetectorResult> {
        // Output shape: [84, 8400]
        // 84 = 4 bbox coords (cx, cy, w, h) + 80 class scores
        // coords are relative to 640x640 input size
        val numClasses = 80
        val results = mutableListOf<DetectorResult>()

        if (output.size < 84) {
            android.util.Log.e("YOLOv8", "输出尺寸错误: ${output.size} < 84")
            return emptyList()
        }

        for (i in 0 until 8400) {
            // cx, cy, w, h - in [0, 640] range (letterbox后的坐标)
            val cx = output[0][i]
            val cy = output[1][i]
            val w = output[2][i]
            val h = output[3][i]

            // Find max class score
            var maxClassScore = 0f
            var maxClassId = 0

            for (c in 0 until numClasses) {
                val score = output[4 + c][i]
                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassId = c
                }
            }

            if (maxClassScore > confThreshold) {

                // 先减去letterbox的padding，再缩放到原始图像尺寸
                val x1 = (cx - w / 2 - padX) / scale
                val y1 = (cy - h / 2 - padY) / scale
                val x2 = (cx + w / 2 - padX) / scale
                val y2 = (cy + h / 2 - padY) / scale

                results.add(
                    DetectorResult(
                        classId = maxClassId,
                        className = classNames[maxClassId],
                        confidence = maxClassScore,
                        x1 = x1,
                        y1 = y1,
                        x2 = x2,
                        y2 = y2
                    )
                )
            }
        }

        return NMSUtils.nonMaxSuppression(results, iouThreshold)
    }

    override fun close() {
        try {
            ortSession?.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        ortSession = null
        // 注意: ortEnvironment 是全局单例，不要关闭
    }
}
