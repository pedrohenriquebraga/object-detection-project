package com.eitalab.objectdetection.src.tensorflow

import android.graphics.Bitmap
import android.util.Log
import com.eitalab.objectdetection.ui.DetectionResult
import org.tensorflow.lite.InterpreterApi as TfInterpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.DataType
import java.io.File
import kotlin.math.abs
import kotlin.math.log

class TensorflowController(private var modelFile: File) {

    private lateinit var interpreter: TfInterpreter
    private var started = false

    private var labels = arrayListOf(
        "cama",                // bed
        "ponto de ônibus",     // bus_stops
        "gatos",               // cats
        "cadeiras",            // chairs
        "armários",            // closets
        "sofás",               // couches
        "faixas de pedestre",  // crosswalks
        "cães",                // dogs
        "portas",              // doors
        "geladeiras",          // fridges
        //"null",                // null (mantido como placeholder)
        "pessoas",             // persons
        "escadas",             // stairs
        "mesas",               // tables
        "árvores",             // trees
        "tv",                  // tv
        "veículos"             // vehicles
    )

    private val maxConf = 0.8f
    private val minDeltaConf = 0.20f

    private val INPUT_SIZE = 320
    private val NUM_CLASSES = labels.size

    init {
        initTensorflow()
    }

    private fun initTensorflow() {
        val options = TfInterpreter.Options().apply {
            setRuntime(TfInterpreter.Options.TfLiteRuntime.PREFER_SYSTEM_OVER_APPLICATION)
            setNumThreads(4)
        }
        try {
            interpreter = TfInterpreter.create(modelFile, options)
            started = true
            Log.i("Tensorflow Controller", "Modelo EfficientNet carregado com sucesso")
        } catch (e: Exception) {
            Log.e("Tensorflow Controller", "Erro ao iniciar Tensorflow: ${e.message}")
        }
    }

    fun detect(imgInput: Bitmap): MutableList<DetectionResult>? {
        if (!started) return null

        val tensorImage = processImage(imgInput)
        val outputProbability = Array(1) { FloatArray(NUM_CLASSES) }

        try {
            interpreter.run(tensorImage.buffer, outputProbability)
        } catch (e: Exception) {
            Log.e("Tensorflow Controller", "Erro na inferência: ${e.message}")
            return null
        }

        val detections = mutableListOf<DetectionResult>()
        val probabilities = outputProbability[0]

        for (i in probabilities.indices) {
            val score = probabilities[i]
            if (score >= maxConf) {
                val label = labels[i]
                if (label != "null") {
                    detections.add(DetectionResult(label, score))
                }
            }
        }

        detections.sortByDescending { it.confidence }

        if (detections.size >= 2) {
            val first = detections[0]
            val second = detections[1]
            val deltaConfidence = abs(first.confidence - second.confidence)
            Log.d("delta", deltaConfidence.toString())

            if (deltaConfidence >= minDeltaConf) {
                return detections
            }
        } else if (detections.size == 1) {
            return detections
        }

        return mutableListOf()
    }

    private fun processImage(bitmap: Bitmap): TensorImage {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        return imageProcessor.process(tensorImage)
    }

    fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        val matrix = android.graphics.Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}