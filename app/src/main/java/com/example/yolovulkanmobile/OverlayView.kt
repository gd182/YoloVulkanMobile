package com.example.yolovulkanmobile

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private var detections: List<Detection> = emptyList()
    private var frameWidth = 0
    private var frameHeight = 0

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.rgb(0, 220, 120)
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 40f
        isFakeBoldText = true
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 220, 120)
        style = Paint.Style.FILL
    }

    fun setResults(detections: List<Detection>, frameWidth: Int, frameHeight: Int) {
        this.detections = detections
        this.frameWidth = frameWidth
        this.frameHeight = frameHeight
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (frameWidth == 0 || frameHeight == 0) return

        val scale = max(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val dx = (width - frameWidth * scale) / 2f
        val dy = (height - frameHeight * scale) / 2f

        val mapped = RectF()
        for (d in detections) {
            mapped.set(
                d.rect.left * scale + dx,
                d.rect.top * scale + dy,
                d.rect.right * scale + dx,
                d.rect.bottom * scale + dy,
            )
            canvas.drawRect(mapped, boxPaint)

            val text = "%s %.0f%%".format(d.labelName, d.score * 100)
            val tw = textPaint.measureText(text)
            val th = textPaint.textSize + 8f
            var top = mapped.top - th
            if (top < 0f) top = mapped.top
            canvas.drawRect(mapped.left, top, mapped.left + tw + 12f, top + th, labelBgPaint)
            canvas.drawText(text, mapped.left + 6f, top + textPaint.textSize, textPaint)
        }
    }
}
