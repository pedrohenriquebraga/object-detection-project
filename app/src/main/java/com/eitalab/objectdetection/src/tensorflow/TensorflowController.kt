package com.eitalab.objectdetection.src.tensorflow
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import org.tensorflow.lite.Interpreter as TfInterpreter
import androidx.core.graphics.scale
import com.eitalab.objectdetection.ui.DetectionResult
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TensorflowController {

    private lateinit var interpreter: TfInterpreter
    private var modelFile: File
    private var LABELS = arrayListOf<String>(
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
    private val max_conf = 0.55f
    private var started = false

    constructor(modelFile: File) {
        this.modelFile = modelFile
    }

    fun initTensorflow() {
        try {
            interpreter = TfInterpreter(modelFile, null)
            interpreter.allocateTensors()
            started = true
            Log.i("Tensorflow Controller","Tensorflow iniciado com sucesso")
        } catch (e: Exception) {
            Log.e("Tensorflow Controller", "Erro ao iniciar Tensorflow: ${e.message}")
        }

    }

    fun detect(imgInput: Bitmap): MutableList<DetectionResult>? {
        if (!started)
            return null

        val imgBuffer = bitmapToTensorImage(imgInput);

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
            if (score >= max_conf) {
                val clsIndex = classes[0][i].toInt()
                val label = if (clsIndex in LABELS.indices) LABELS[clsIndex] else "Desconhecido"
                val box = boxes[0][i]
                detections.add(DetectionResult(label, score, box))
            }
        }

        return detections
    }

    private fun bitmapToTensorImage(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = bitmap.scale(320, 320)
        val tensorImage = TensorImage.fromBitmap(resizedBitmap)
        val buffer = tensorImage.buffer
        buffer.order(ByteOrder.nativeOrder())
        return buffer
    }

}