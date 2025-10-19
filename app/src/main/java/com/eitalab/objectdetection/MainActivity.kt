package com.eitalab.objectdetection

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.graphics.get
import androidx.core.graphics.toColor
import androidx.lifecycle.LifecycleOwner
import com.eitalab.objectdetection.databinding.MainActivityBinding
import com.eitalab.objectdetection.src.tensorflow.TensorflowController
import com.google.common.util.concurrent.ListenableFuture
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var binding: MainActivityBinding
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraProvider: ProcessCameraProvider
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private lateinit var modelFile: File;
    private lateinit var tf: TensorflowController
    private lateinit var capturedImage: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = MainActivityBinding.inflate(layoutInflater)

        setContentView(binding.root)
        enableEdgeToEdge()

        modelFile = getModelFileFromAssets("efficientdet-lite0.tflite")
        tf = TensorflowController(modelFile)
        tf.initTensorflow()

        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindPreview()
        }, ContextCompat.getMainExecutor(this))

    }

    fun bindPreview() {
        val preview : Preview = Preview.Builder()
            .build()

        val cameraSelector : CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()


        preview.surfaceProvider = binding.cameraPreview.getSurfaceProvider()

        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    capturedImage = capturedImageToBitmap(imageProxy)
                    val detections = tf.detect(capturedImage)
                    if (detections != null) {
                        runOnUiThread {
                            binding.detectionOverlay.setResults(detections)
                        }
                    }
                }
            }

        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview, imageAnalyzer)
        //binding.btDetect.setOnClickListener {
        //    tf.detect(capturedImage)
        //}
    }

    private fun capturedImageToBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()
        imageProxy.close()
        return bitmap
    }

    private fun getModelFileFromAssets(fileName: String): File {
        val file = File(filesDir, fileName)
        if (file.exists()) {
            return file
        }

        try {
            assets.open(fileName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "Error copying model from assets", e)
        }
        return file
    }
}
