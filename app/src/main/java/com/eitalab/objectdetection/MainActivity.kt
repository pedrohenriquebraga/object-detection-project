package com.eitalab.objectdetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Size
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.eitalab.objectdetection.databinding.MainActivityBinding
import com.eitalab.objectdetection.src.tensorflow.TensorflowController
import com.eitalab.objectdetection.src.tensorflow.Utils
import com.google.common.util.concurrent.ListenableFuture
import java.io.File
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private val CAMERA_PERMISSION_REQUEST_CODE = 100
    private lateinit var tf: TensorflowController
    private val utils = Utils()
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private lateinit var binding: MainActivityBinding
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var modelFile: File;
    private lateinit var capturedImage: Bitmap
    private var cameraGranted = false;
    private var lastAnalyzedTime = 0L

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE)
        } else {
            cameraGranted = true
            initCamera()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int,
                                            permissions: Array<String>,
                                            grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST_CODE -> {
                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    cameraGranted = true
                    initCamera()
                } else {
                    Toast.makeText(this, "Autorize o uso da câmera", Toast.LENGTH_LONG).show()
                }
            }
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = MainActivityBinding.inflate(layoutInflater)

        setContentView(binding.root)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        requestCameraPermission()

    }

    fun initCamera() {
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
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastAnalyzedTime >= 250) {
                        capturedImage = utils.capturedImageToBitmap(imageProxy, tf)
                        val detections = tf.detect(capturedImage)
                        if (detections != null) {
                            runOnUiThread {
                                binding.detectionOverlay.setResults(detections, 320, 320)
                            }
                        }
                    } else {
                        imageProxy.close()
                    }
                }
            }

        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview, imageAnalyzer)
        //binding.btDetect.setOnClickListener {
        //    tf.detect(capturedImage)
        //}
    }
}
