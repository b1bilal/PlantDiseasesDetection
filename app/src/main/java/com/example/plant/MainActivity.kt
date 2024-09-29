package com.example.plant

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var btnClassification: Button
    private lateinit var btnRealTimeDetection: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnClassification = findViewById(R.id.btnClassification)
        btnRealTimeDetection = findViewById(R.id.btnRealTimeDetection)

        btnClassification.setOnClickListener {
            val intent = Intent(this, ClassificationActivity::class.java)
            startActivity(intent)
        }

        btnRealTimeDetection.setOnClickListener {
            val intent = Intent(this, RealTimeDetectionActivity::class.java)
            startActivity(intent)
        }
    }
}


