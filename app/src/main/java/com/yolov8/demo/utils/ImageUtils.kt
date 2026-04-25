package com.yolov8.demo.utils

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.renderscript.Type
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImageUtils {

    private var rs: RenderScript? = null
    private var yuvToRgb: ScriptIntrinsicYuvToRGB? = null

    // 暂时不需要RenderScript初始化
    fun initRenderScript(context: android.content.Context) {
    }

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
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 90, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun yuvToBitmapRenderscript(yuv: ByteArray, width: Int, height: Int): Bitmap {
        val yuvType = Type.Builder(rs, Element.U8(rs)).setX(yuv.size).create()
        val yuvAllocation = Allocation.createTyped(rs, yuvType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_SCRIPT)
        yuvAllocation.copyFrom(yuv)

        val bitmapType = Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height).create()
        val rgbAllocation = Allocation.createTyped(rs, bitmapType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_SCRIPT)

        yuvToRgb!!.setInput(yuvAllocation)
        yuvToRgb!!.forEach(rgbAllocation)

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        rgbAllocation.copyTo(bitmap)
        return bitmap
    }

    fun resizeBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }

    data class PreprocessResult(
        val floatArray: FloatArray,
        val scale: Float,
        val padX: Float,
        val padY: Float
    )

    fun bitmapToFloatBuffer(bitmap: Bitmap, inputSize: Int): PreprocessResult {
        // Letterbox: 保持宽高比，填充黑边
        val scale = minOf(inputSize.toFloat() / bitmap.width, inputSize.toFloat() / bitmap.height)
        val scaledW = (bitmap.width * scale).toInt()
        val scaledH = (bitmap.height * scale).toInt()
        val padX = (inputSize - scaledW) / 2f
        val padY = (inputSize - scaledH) / 2f

        // Resize bitmap
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)

        // Create result bitmap with padding
        val resultBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(resultBitmap)
        canvas.drawColor(android.graphics.Color.BLACK)
        canvas.drawBitmap(scaledBitmap, padX, padY, null)

        // Convert to float array
        val floatArray = FloatArray(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        resultBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val area = inputSize * inputSize

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            floatArray[i] = r
            floatArray[i + area] = g
            floatArray[i + 2 * area] = b
        }

        scaledBitmap.recycle()
        resultBitmap.recycle()

        return PreprocessResult(floatArray, scale, padX, padY)
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
