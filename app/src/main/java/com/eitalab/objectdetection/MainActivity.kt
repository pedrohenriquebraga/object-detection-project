package com.eitalab.objectdetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.eitalab.objectdetection.databinding.MainActivityBinding
import com.eitalab.objectdetection.src.services.BluetoothService
import com.eitalab.objectdetection.src.tensorflow.TensorflowController
import com.eitalab.objectdetection.src.tensorflow.Utils
import com.google.common.util.concurrent.ListenableFuture
import java.io.File
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var tf: TensorflowController
    private lateinit var bluetoothService: BluetoothService
    private val utils = Utils()
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private lateinit var binding: MainActivityBinding
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var modelFile: File
    private lateinit var capturedImage: Bitmap

    private var isBluetoothInitialized = false
    private var lastAnalyzedTime = 0L

    private val requestMultiplePermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val allGranted = permissions.entries.all { it.value }

            if (allGranted) {
                startAppFeatures()
            } else {
                Toast.makeText(
                    this,
                    "Todas as permissões (Câmera, Bluetooth e Localização) são necessárias.",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)
        enableEdgeToEdge()
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        checkAndRequestPermissions()
    }

    private fun checkAndRequestPermissions() {
        val permissionsToRequest = getRequiredPermissions()

        val allPermissionsGranted = permissionsToRequest.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

        if (allPermissionsGranted) {
            startAppFeatures()
        } else {
            requestMultiplePermissionsLauncher.launch(permissionsToRequest)
        }
    }

    private fun getRequiredPermissions(): Array<String> {
        val permissions = mutableListOf(Manifest.permission.CAMERA)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions.add(Manifest.permission.BLUETOOTH_SCAN)
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
            permissions.add(Manifest.permission.ACCESS_FINE_LOCATION)
            permissions.add(Manifest.permission.ACCESS_COARSE_LOCATION)
        }
        return permissions.toTypedArray()
    }

    private fun startAppFeatures() {
        initCamera()
        initBluetoothService()
    }

    private fun initCamera() {
        modelFile = utils.getModelFileFromAssets("efficientdet-lite0.tflite", filesDir, assets)
        tf = TensorflowController(modelFile)

        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindPreview()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun initBluetoothService() {
        if (isBluetoothInitialized) return

        bluetoothService = BluetoothService(this)
        bluetoothService.turnOnBluetooth()

        initAssistiveDeviceConnection()
        isBluetoothInitialized = true
    }

    private fun initAssistiveDeviceConnection() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.BLUETOOTH_SCAN
                ) != PackageManager.PERMISSION_GRANTED
            ) return
        } else {
            if (ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_FINE_LOCATION
                ) != PackageManager.PERMISSION_GRANTED
            ) return
        }

        bluetoothService.scanLeDevice(object : BluetoothService.OnDeviceFoundListener {
            override fun onDeviceFound(device: android.bluetooth.BluetoothDevice) {
                try {
                    Log.d("SCAN", "Achou: ${device.name ?: "Sem Nome"} - ${device.address}")
                    if (device.name == "BT05") {
                        bluetoothService.connectAssistiveDevice(device.address)
                    }
                } catch (e: SecurityException) {
                    Log.e(
                        "Bluetooth",
                        "Erro de permissão ao acessar nome do dispositivo: ${e.message}"
                    )
                }
            }

            override fun onScanFinished() {
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Scan finalizado", Toast.LENGTH_SHORT).show()
                }
            }
        })
    }

    private fun bindPreview() {
        val preview: Preview = Preview.Builder().build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.surfaceProvider = binding.cameraPreview.surfaceProvider

        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastAnalyzedTime >= 250) {
                        lastAnalyzedTime = currentTime
                        capturedImage = utils.capturedImageToBitmap(imageProxy, tf)
                        val detections = tf.detect(capturedImage)

                        if (detections != null) {
                            runOnUiThread {
                                binding.detectionOverlay.setResults(detections, 320, 320)
                                detections.forEach { detect ->
                                    if (::bluetoothService.isInitialized) {
                                        bluetoothService.sendMessage("Objeto ${detect.label} detectado com ${detect.confidence}\n")
                                    }
                                }
                            }
                        }
                        imageProxy.close()
                    } else {
                        imageProxy.close()
                    }
                }
            }

        cameraProvider.bindToLifecycle(
            this as LifecycleOwner,
            cameraSelector,
            preview,
            imageAnalyzer
        )
    }
}