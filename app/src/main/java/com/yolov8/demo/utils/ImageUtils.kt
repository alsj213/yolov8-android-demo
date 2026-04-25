package com.yolov8.demo.utils

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImageUtils {

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(
            nv21,
            android.graphics.ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    fun resizeBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }

    fun bitmapToFloatBuffer(bitmap: Bitmap, inputSize: Int): FloatArray {
        val floatArray = FloatArray(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f
            floatArray[i + inputSize * inputSize] = ((pixel shr 8) and 0xFF) / 255.0f
            floatArray[i + 2 * inputSize * inputSize] = (pixel and 0xFF) / 255.0f
        }
        return floatArray
    }

    fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    fun createDirectFloatBuffer(floatArray: FloatArray): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(floatArray.size * 4)
        buffer.order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(floatArray)
        buffer.rewind()
        return buffer
    }
}
