package com.yolov8.demo.detector

import android.content.Context
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
    private val numThreads: Int = 4
) {
    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null

    val inputSize = 640
    val confThreshold = 0.25f
    val iouThreshold = 0.45f

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

    suspend fun init() = withContext(Dispatchers.IO) {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(numThreads)
            sessionOptions.setInterOpNumThreads(2)

            // Enable NNAPI if requested
            if (useNNAPI) {
                sessionOptions.addConfigEntry("session.use_nnapi", "1")
                sessionOptions.addConfigEntry("session.nnapi.use_fp16", "1")
            }

            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            val modelFile = getModelFile()
            ortSession = ortEnvironment!!.createSession(modelFile.absolutePath, sessionOptions)

        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("Failed to initialize ONNX Runtime", e)
        }
    }

    private fun getModelFile(): File {
        val modelFileName = "yolov8n.onnx"
        val outputFile = File(context.filesDir, modelFileName)

        if (!outputFile.exists()) {
            context.resources.openRawResource(
                context.resources.getIdentifier("yolov8n", "raw", context.packageName)
            ).use { input ->
                FileOutputStream(outputFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outputFile
    }

    suspend fun detect(bitmap: android.graphics.Bitmap): DetectionResult = withContext(Dispatchers.Default) {
        val startTime = SystemClock.elapsedRealtimeNanos()

        val resizedBitmap = ImageUtils.resizeBitmap(bitmap, inputSize, inputSize)
        val floatArray = ImageUtils.bitmapToFloatBuffer(resizedBitmap, inputSize)

        val inputName = ortSession!!.inputNames.first()
        val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())

        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(floatArray),
            inputShape
        )

        val output = ortSession!!.run(mapOf(inputName to inputTensor))
        val outputTensor = output.get(0).value as Array<*>

        val results = parseOutput(outputTensor, bitmap.width, bitmap.height)

        val inferenceTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000f

        output.close()
        inputTensor.close()

        DetectionResult(results, inferenceTime)
    }

    private fun parseOutput(output: Array<*>, imgWidth: Int, imgHeight: Int): List<DetectorResult> {
        val predictions = output[0] as Array<*>
        val numClasses = 80
        val numDetections = predictions[0] as? FloatArray ?: return emptyList()

        val results = mutableListOf<DetectorResult>()

        // Output shape: [1, 84, 8400] - 84 = 4 bbox coords + 80 class scores
        for (i in 0 until 8400) {
            val x = (predictions[0] as FloatArray)[i]
            val y = (predictions[1] as FloatArray)[i]
            val w = (predictions[2] as FloatArray)[i]
            val h = (predictions[3] as FloatArray)[i]

            var maxClassScore = 0f
            var maxClassId = 0

            for (c in 0 until numClasses) {
                val score = (predictions[4 + c] as FloatArray)[i]
                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassId = c
                }
            }

            if (maxClassScore > confThreshold) {
                val scaleX = imgWidth / inputSize.toFloat()
                val scaleY = imgHeight / inputSize.toFloat()

                val x1 = (x - w / 2) * scaleX
                val y1 = (y - h / 2) * scaleY
                val x2 = (x + w / 2) * scaleX
                val y2 = (y + h / 2) * scaleY

                results.add(
                    DetectorResult(
                        classId = maxClassId,
                        className = classNames[maxClassId],
                        confidence = maxClassScore,
                        x1 = x1.coerceIn(0f, imgWidth.toFloat()),
                        y1 = y1.coerceIn(0f, imgHeight.toFloat()),
                        x2 = x2.coerceIn(0f, imgWidth.toFloat()),
                        y2 = y2.coerceIn(0f, imgHeight.toFloat())
                    )
                )
            }
        }

        return NMSUtils.nonMaxSuppression(results, iouThreshold)
    }

    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
    }

    data class DetectionResult(
        val results: List<DetectorResult>,
        val inferenceTimeMs: Float
    )
}
