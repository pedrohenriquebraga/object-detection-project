package com.eitalab.objectdetection.src.tensorflow
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import org.tensorflow.lite.InterpreterApi as TfInterpreter
import androidx.core.graphics.scale
import com.eitalab.objectdetection.ui.DetectionResult
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TensorflowController {

    private lateinit var interpreter: TfInterpreter
    private var modelFile: File
    private var labels = arrayListOf<String>(
        "pessoa",
        "bicicleta",
        "carro",
        "moto",
        "aviao",
        "onibus",
        "trem",
        "caminhao",
        "barco",
        "semaforo",
        "hidrante",
        "sinal de transito",
        "placa de pare",
        "parquimetro",
        "banco",
        "passaro",
        "gato",
        "cachorro",
        "cavalo",
        "ovelha",
        "vaca",
        "elefante",
        "urso",
        "zebra",
        "girafa",
        "chapeu",
        "mochila",
        "guarda-chuva",
        "sapato",
        "oculos",
        "bolsa",
        "gravata",
        "mala",
        "frisbee",
        "esquis",
        "snowboard",
        "bola de esporte",
        "pipa",
        "taco de beisebol",
        "luva de beisebol",
        "skate",
        "prancha de surfe",
        "Raquete de tenis",
        "garrafa",
        "prato",
        "taca de vinho",
        "copo",
        "garfo",
        "faca",
        "colher",
        "tigela",
        "banana",
        "maca",
        "sanduiche",
        "laranja",
        "brocolis",
        "cenoura",
        "cachorro-quente",
        "pizza",
        "donut",
        "bolo",
        "cadeira",
        "sofa",
        "planta em vaso",
        "cama",
        "espelho",
        "mesa de jantar",
        "janela",
        "mesa",
        "vaso sanitario",
        "porta",
        "TV",
        "laptop",
        "mouse",
        "controle remoto",
        "teclado",
        "celular",
        "micro-ondas",
        "forno",
        "torradeira",
        "pia",
        "geladeira",
        "liquidificador",
        "livro",
        "relogio",
        "vaso",
        "tesoura",
        "ursinho de pelucia",
        "secador de cabelo",
        "escova de dentes",
        "escova de cabelo"
    )
    private val maxConf = 0.55f
    private var started = false

    constructor(modelFile: File) {
        this.modelFile = modelFile
        initTensorflow()
    }

    fun initTensorflow() {

        val options = TfInterpreter.Options().apply {
            setRuntime(TfInterpreter.Options.TfLiteRuntime.PREFER_SYSTEM_OVER_APPLICATION)
        }
        try {
            interpreter = TfInterpreter.create(modelFile, options)
            started = true
            Log.i("Tensorflow Controller","Tensorflow iniciado com sucesso")
        } catch (e: Exception) {
            Log.e("Tensorflow Controller", "Erro ao iniciar Tensorflow: ${e.message}")
        }
    }

    fun detect(imgInput: Bitmap): MutableList<DetectionResult>? {
        if (!started)
            return null

        val imgBuffer = bitmapToTensorImage(imgInput, 320, 320);

        val boxes = Array(1) { Array(25) { FloatArray(4) } }
        val classes = Array(1) { FloatArray(25) }
        val scores = Array(1) { FloatArray(25) }

        val outputs = mutableMapOf<Int, Any>(
            0 to boxes,
            1 to classes,
            2 to scores,
        )

        interpreter.runForMultipleInputsOutputs(arrayOf(imgBuffer), outputs)

        val numDetections = scores[0].size
        val detections = mutableListOf<DetectionResult>()

        for (i in 0 until numDetections) {
            val score = scores[0][i]
            if (score >= maxConf) {
                val clsIndex = classes[0][i].toInt()
                val label = if (clsIndex in labels.indices) labels[clsIndex] else "Desconhecido"
                val box = boxes[0][i]
                detections.add(DetectionResult(label, score, box))
            }
        }

        return detections
    }

    private fun bitmapToTensorImage(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        // TODO: CROP IMAGE
        val utils = Utils()
        val resizedBitmap = utils.cropCenter(bitmap, width, height)
        val tensorImage = TensorImage.fromBitmap(resizedBitmap)
        val buffer = tensorImage.buffer
        buffer.order(ByteOrder.nativeOrder())
        return buffer
    }

    fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        val matrix = android.graphics.Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }


}