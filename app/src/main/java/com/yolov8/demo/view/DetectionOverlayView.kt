package com.yolov8.demo.view

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.yolov8.demo.detector.DetectorResult
import kotlin.math.min

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint().apply {
        color = Color.parseColor("#4CAF50")
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        color = Color.parseColor("#CC000000")
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        isAntiAlias = true
    }

    private var results: List<DetectorResult> = emptyList()
    private var scaleX = 1f
    private var scaleY = 1f

    fun setResults(results: List<DetectorResult>, imageWidth: Int, imageHeight: Int) {
        this.results = results
        scaleX = width / imageWidth.toFloat()
        scaleY = height / imageHeight.toFloat()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (result in results) {
            val x1 = result.x1 * scaleX
            val y1 = result.y1 * scaleY
            val x2 = result.x2 * scaleX
            val y2 = result.y2 * scaleY

            // Draw bounding box
            canvas.drawRect(x1, y1, x2, y2, boxPaint)

            // Draw label background
            val label = "${result.className} ${String.format("%.2f", result.confidence)}"
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize

            canvas.drawRect(
                x1,
                y1 - textHeight - 8f,
                x1 + textWidth + 16f,
                y1,
                textBgPaint
            )

            // Draw label text
            canvas.drawText(label, x1 + 8f, y1 - 8f, textPaint)
        }
    }
}
