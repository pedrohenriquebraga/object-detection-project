package com.eitalab.objectdetection

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.eitalab.objectdetection.databinding.MainActivityBinding
import com.eitalab.objectdetection.src.tensorflow.TensorflowController
import com.eitalab.objectdetection.src.tensorflow.Utils
import com.google.common.util.concurrent.ListenableFuture
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var tf: TensorflowController
    private val utils = Utils()
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private lateinit var binding: MainActivityBinding
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var modelFile: File;
    private lateinit var capturedImage: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = MainActivityBinding.inflate(layoutInflater)

        setContentView(binding.root)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        modelFile = utils.getModelFileFromAssets("efficientdet-lite0.tflite", filesDir, assets)
        tf = TensorflowController(modelFile)

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
                    capturedImage = utils.capturedImageToBitmap(imageProxy, tf)
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
}
