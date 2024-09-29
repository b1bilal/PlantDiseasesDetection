

package com.example.plant

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import org.tensorflow.lite.support.common.FileUtil
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class RealTimeDetectionActivity : AppCompatActivity() {

    private lateinit var tvDisease: TextView
    private lateinit var tvSolution: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var categories: Map<Int, Pair<String, String>>
    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_real_time_detection)

        tvDisease = findViewById(R.id.tvDisease)
        tvSolution = findViewById(R.id.tvSolution)

        cameraExecutor = Executors.newSingleThreadExecutor()

        loadModelAndLabels()
        startCamera()
    }

    private fun loadModelAndLabels() {
        // TensorFlow Lite modelini yükleyin
        val modelFile = FileUtil.loadMappedFile(this, "model.tflite")
        interpreter = Interpreter(modelFile)

        // JSON dosyasını okuyun
        val categoriesJson = loadJSONFromAsset("categories.json")
        categories = parseCategoriesJson(categoriesJson)
    }

    private fun loadJSONFromAsset(fileName: String): JSONObject {
        val json: String?
        try {
            val inputStream = assets.open(fileName)
            val size = inputStream.available()
            val buffer = ByteArray(size)
            inputStream.read(buffer)
            inputStream.close()
            json = String(buffer, Charsets.UTF_8)
        } catch (ex: Exception) {
            ex.printStackTrace()
            return JSONObject()
        }
        return JSONObject(json)
    }

    private fun parseCategoriesJson(jsonObject: JSONObject): Map<Int, Pair<String, String>> {
        val categories = mutableMapOf<Int, Pair<String, String>>()
        jsonObject.keys().forEach { key ->
            val categoryJson = jsonObject.getJSONObject(key)
            val disease = categoryJson.getString("Hastalık")
            val solution = categoryJson.getString("Çözüm")
            categories[key.toInt()] = Pair(disease, solution)
        }
        return categories
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { imageProxy ->
                        processImageProxy(imageProxy)
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (exc: Exception) {
                Toast.makeText(this, "Kamera bağlanamadı", Toast.LENGTH_SHORT).show()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    @OptIn(ExperimentalGetImage::class)
    private fun processImageProxy(imageProxy: ImageProxy) {
        if (imageProxy.image != null) {
            val mediaImage = imageProxy.image
            val bitmap = mediaImage?.toBitmap()
            if (bitmap != null) {
                val (disease, solution) = classifyImage(bitmap)
                runOnUiThread {
                    tvDisease.text = disease
                    tvSolution.text = solution
                }
            }
        }
        imageProxy.close()
    }


    private fun Image.toBitmap(): Bitmap? {
        val nv21 = yuv420888ToNv21(this)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun yuv420888ToNv21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4

        val nv21 = ByteArray(ySize + uvSize * 2)

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        var rowStride = image.planes[0].rowStride
        var pos = 0

        if (rowStride == width) {
            yBuffer[nv21, 0, ySize]
            pos += ySize
        } else {
            var yBufferPos = -rowStride
            while (pos < ySize) {
                yBufferPos += rowStride
                yBuffer.position(yBufferPos)
                yBuffer[nv21, pos, width]
                pos += width
            }
        }

        rowStride = image.planes[2].rowStride
        val pixelStride = image.planes[2].pixelStride

        if (pixelStride == 2 && rowStride == width) {
            vBuffer[nv21, ySize, uvSize]
            uBuffer[nv21, ySize + uvSize, uvSize]
        } else {
            for (i in 0 until height / 2) {
                for (j in 0 until width / 2) {
                    nv21[ySize + i * width + j * 2] = vBuffer[i * rowStride + j * pixelStride]
                    nv21[ySize + i * width + j * 2 + 1] = uBuffer[i * rowStride + j * pixelStride]
                }
            }
        }
        return nv21
    }

    private fun classifyImage(bitmap: Bitmap): Pair<String, String> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3).apply { order(ByteOrder.nativeOrder()) }

        val intValues = IntArray(224 * 224)
        resizedBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)
        for (pixel in intValues) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
        }

        val outputBuffer = ByteBuffer.allocateDirect(4 * categories.size).apply { order(ByteOrder.nativeOrder()) }
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        val probabilities = FloatArray(categories.size)
        outputBuffer.asFloatBuffer().get(probabilities)

        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
        return if (maxIndex != -1 && maxIndex in categories) {
            categories[maxIndex] ?: "Bilinmiyor" to "çözüm yok"
        } else {
            "Bilinmiyor" to "çözüm yok"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

