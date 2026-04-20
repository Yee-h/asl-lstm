package com.ye.asl_lstm;

import android.content.Intent;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findViewById(R.id.btn_realtime).setOnClickListener(v -> {
            startActivity(new Intent(MainActivity.this, RealtimeActivity.class));
        });

        findViewById(R.id.btn_offline).setOnClickListener(v -> {
            startActivity(new Intent(MainActivity.this, OfflineActivity.class));
        });
    }
}