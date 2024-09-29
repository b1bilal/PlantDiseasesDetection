package com.example.plant

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class ClassificationActivity : AppCompatActivity() {

    // Sınıf değişkenlerinin tanımlanması
    private lateinit var categories: Map<Int, Pair<String, String>> // Hastalık kategorilerini ve çözümlerini içeren bir harita
    private lateinit var interpreter: Interpreter // TensorFlow Lite modelini yorumlamak için bir yorumlayıcı
    private lateinit var btnSelectImage: Button // Resim seçmek için bir düğme
    private lateinit var imageView: ImageView // Seçilen resmi gösteren bir görüntü
    private lateinit var tvDisease: TextView // Tanı konulan hastalığı gösteren bir metin görüntüsü
    private lateinit var tvSolution: TextView // Hastalık için önerilen çözümü gösteren bir metin görüntüsü
    private val SELECT_PHOTO = 1 // Resim seçmek için kullanılan isteğin kodu

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_classification)

        // Arayüz öğelerinin başlatılması
        btnSelectImage = findViewById(R.id.btnSelectImage)
        imageView = findViewById(R.id.imageView)
        tvDisease = findViewById(R.id.tvDisease)
        tvSolution = findViewById(R.id.tvSolution)

        // Model ve etiketlerin yüklenmesi
        loadModelAndLabels()

        // Resim seçmek için düğmenin tıklanma işlevi
        btnSelectImage.setOnClickListener {
            val photoPickerIntent = Intent(Intent.ACTION_PICK)
            photoPickerIntent.type = "image/*"
            startActivityForResult(photoPickerIntent, SELECT_PHOTO)
        }
    }

    private fun loadModelAndLabels() {
        try {
            val assetFileDescriptor = this.assets.openFd("model.tflite")
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            interpreter = Interpreter(modelBuffer)

            categories = loadLabelList()
        } catch (e: IOException) {
            print("Hata")
        }
    }

    private fun loadLabelList(): Map<Int, Pair<String, String>> {
        val labelsInputStream = assets.open("categories.json")
        val size = labelsInputStream.available()
        val buffer = ByteArray(size)
        labelsInputStream.read(buffer)
        labelsInputStream.close()
        val json = String(buffer, Charsets.UTF_8)
        val jsonObject = JSONObject(json)
        val labelsMap = mutableMapOf<Int, Pair<String, String>>()
        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val diseaseInfo = jsonObject.getJSONObject(key)
            val disease = diseaseInfo.getString("Hastalık")
            val solution = diseaseInfo.getString("Çözüm")
            labelsMap[key.toInt()] = Pair(disease, solution)
        }
        return labelsMap
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == SELECT_PHOTO && resultCode == Activity.RESULT_OK && data != null) {
            val imageUri = data.data
            val imageStream = contentResolver.openInputStream(imageUri!!)
            val selectedImage = BitmapFactory.decodeStream(imageStream)
            imageView.setImageBitmap(selectedImage)

            val (disease, solution) = classifyImage(selectedImage)
            tvDisease.text = "$disease"
            tvSolution.text = "$solution"
        }
    }

    private fun classifyImage(bitmap: Bitmap): Pair<String, String> {
        val resizedImage = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val byteBuffer = convertBitmapToByteBuffer(resizedImage)
        val result = Array(1) { FloatArray(categories.size) }
        interpreter.run(byteBuffer, result)
        return getLabel(result[0])
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(224 * 224)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat((`val` shr 16 and 0xFF) / 255f)
                byteBuffer.putFloat((`val` shr 8 and 0xFF) / 255f)
                byteBuffer.putFloat((`val` and 0xFF) / 255f)
            }
        }
        return byteBuffer
    }

    private fun getLabel(probability: FloatArray): Pair<String, String> {
        var maxProb = 0f
        var index = -1
        for (i in probability.indices) {
            if (probability[i] > maxProb) {
                maxProb = probability[i]
                index = i
            }
        }
        return categories[index] ?: Pair("Hastalık bulunamadı", "Çözüm bulunamadı")
    }
}
