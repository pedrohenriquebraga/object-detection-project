package com.eitalab.objectdetection.src.tensorflow

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class Utils {
     fun capturedImageToBitmap(imageProxy: ImageProxy, tf: TensorflowController): Bitmap {
        val bitmap = imageProxy.toBitmap()
        val rotatedBitmap = tf.rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
        imageProxy.close()
        return rotatedBitmap
    }

    fun getModelFileFromAssets(fileName: String, filesDir: File, assets: AssetManager): File {
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