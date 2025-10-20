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

    fun setResults(detections: List<DetectionResult>) {
        results = detections
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        for (det in results) {
            val box = det.box
            val left = box[1] * w
            val top = box[0] * h
            val right = box[3] * w
            val bottom = box[2] * h

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val labelText = "${det.label} ${(det.confidence * 100).toInt()}%"
            canvas.drawText(labelText, left + 10, top + 40, textPaint)
        }
    }
}
