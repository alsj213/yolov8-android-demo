package com.yolov8.demo.detector

import android.graphics.Bitmap

/**
 * 检测器通用接口
 */
interface Detector {

    /**
     * 初始化模型
     */
    suspend fun init()

    /**
     * 执行检测
     */
    suspend fun detect(bitmap: Bitmap): DetectionResult

    /**
     * 释放资源
     */
    fun close()

    /**
     * 获取输入尺寸
     */
    val inputSize: Int

    /**
     * 获取后端名称
     */
    val backendName: String
}

/**
 * 检测结果
 */
data class DetectionResult(
    val detections: List<DetectorResult>,
    val inferenceTimeMs: Float,
    val scale: Float,
    val padX: Float,
    val padY: Float
)
