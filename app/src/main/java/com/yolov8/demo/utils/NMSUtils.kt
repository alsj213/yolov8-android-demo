package com.yolov8.demo.utils

import com.yolov8.demo.detector.DetectorResult
import kotlin.math.max
import kotlin.math.min

object NMSUtils {

    fun nonMaxSuppression(
        results: List<DetectorResult>,
        iouThreshold: Float = 0.45f
    ): List<DetectorResult> {
        if (results.isEmpty()) return emptyList()

        val sortedResults = results.sortedByDescending { it.confidence }
        val selected = mutableListOf<DetectorResult>()

        for (result in sortedResults) {
            var shouldSelect = true

            for (selectedResult in selected) {
                val iou = calculateIoU(result, selectedResult)
                if (iou > iouThreshold) {
                    shouldSelect = false
                    break
                }
            }

            if (shouldSelect) {
                selected.add(result)
            }
        }

        return selected
    }

    private fun calculateIoU(box1: DetectorResult, box2: DetectorResult): Float {
        val x1 = max(box1.x1, box2.x1)
        val y1 = max(box1.y1, box2.y1)
        val x2 = min(box1.x2, box2.x2)
        val y2 = min(box1.y2, box2.y2)

        val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val box1Area = box1.width * box1.height
        val box2Area = box2.width * box2.height

        return intersectionArea / (box1Area + box2Area - intersectionArea + 1e-6f)
    }
}
