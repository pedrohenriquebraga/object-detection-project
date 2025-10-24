package com.eitalab.objectdetection.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import androidx.core.graphics.toColorInt

data class DetectionResult(
    val label: String,
    val confidence: Float,
    val box: FloatArray
)

class DetectionOverlay @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = "#FF3C6E".toColorInt()
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        typeface = Typeface.DEFAULT_BOLD
    }

    private var results: List<DetectionResult> = emptyList()

    private var modelWidth = 320f
    private var modelHeight = 320f

    fun setResults(
        detections: List<DetectionResult>,
        modelWidth: Int,
        modelHeight: Int
    ) {
        this.results = detections
        this.modelWidth = modelWidth.toFloat()
        this.modelHeight = modelHeight.toFloat()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        if (modelWidth == 0f || modelHeight == 0f) return

        val scaleX = viewWidth / modelWidth
        val scaleY = viewHeight / modelHeight
        val scale = minOf(scaleX, scaleY)

        val offsetX = (viewWidth - modelWidth * scale) / 2
        val offsetY = (viewHeight - modelHeight * scale) / 2

        for (det in results) {
            val box = det.box

            val top = box[0] * modelHeight * scale + offsetY
            val left = box[1] * modelWidth * scale + offsetX
            val bottom = box[2] * modelHeight * scale + offsetY
            val right = box[3] * modelWidth * scale + offsetX

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val labelText = "${det.label} ${(det.confidence * 100).toInt()}%"
            canvas.drawText(labelText, left + 10, top + 40, textPaint)
        }
    }
}
