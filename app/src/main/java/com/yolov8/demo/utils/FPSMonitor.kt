package com.yolov8.demo.utils

import android.os.SystemClock
import java.util.LinkedList
import kotlin.math.min

class FPSMonitor(private val windowSize: Int = 30) {
    private val timestamps = LinkedList<Long>()
    private val inferenceTimes = LinkedList<Long>()

    fun update() {
        val now = SystemClock.elapsedRealtime()
        timestamps.addLast(now)
        while (timestamps.size > windowSize) {
            timestamps.removeFirst()
        }
    }

    fun addInferenceTime(timeMs: Long) {
        inferenceTimes.addLast(timeMs)
        while (inferenceTimes.size > windowSize) {
            inferenceTimes.removeFirst()
        }
    }

    fun getFPS(): Float {
        if (timestamps.size < 2) return 0f
        val elapsed = timestamps.last - timestamps.first
        return if (elapsed > 0) {
            (timestamps.size - 1) * 1000f / elapsed
        } else 0f
    }

    fun getAvgInferenceTime(): Float {
        if (inferenceTimes.isEmpty()) return 0f
        return inferenceTimes.average().toFloat()
    }

    fun reset() {
        timestamps.clear()
        inferenceTimes.clear()
    }
}
