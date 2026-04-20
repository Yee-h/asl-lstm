package com.ye.asl_lstm;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RealtimeActivity extends AppCompatActivity implements KeypointExtractor.ExtractorListener, ModelRunner.InferenceListener {
    private static final String TAG = "RealtimeActivity";
    private PreviewView viewFinder;
    private SkeletonOverlayView skeletonOverlay;
    private TextView tvToggleSkeleton;
    private TextView tvConfidence;
    private TextView tvDelay;
    private TextView tvResult;
    private ExecutorService cameraExecutor;
    private KeypointExtractor keypointExtractor;
    private ModelRunner modelRunner;
    private boolean isSkeletonVisible = true;

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startCamera();
                } else {
                    Toast.makeText(this, "需要相机权限才能进行实时识别", Toast.LENGTH_SHORT).show();
                    finish();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_realtime);

        viewFinder = findViewById(R.id.view_finder);
        skeletonOverlay = findViewById(R.id.skeleton_overlay);
        tvToggleSkeleton = findViewById(R.id.tv_toggle_skeleton);
        tvConfidence = findViewById(R.id.tv_confidence);
        tvDelay = findViewById(R.id.tv_delay);
        tvResult = findViewById(R.id.tv_result);

        cameraExecutor = Executors.newSingleThreadExecutor();

        keypointExtractor = new KeypointExtractor(this, this);

        try {
            modelRunner = new ModelRunner(this, "best_model.ptl");
        } catch (Exception e) {
            Toast.makeText(this, "模型加载失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
            Log.e(TAG, "Model load error", e);
        }

        findViewById(R.id.iv_back).setOnClickListener(v -> finish());

        tvToggleSkeleton.setOnClickListener(v -> {
            isSkeletonVisible = !isSkeletonVisible;
            tvToggleSkeleton.setText(isSkeletonVisible ? "隐藏骨骼点" : "显示骨骼点");
            skeletonOverlay.setVisibility(isSkeletonVisible ? android.view.View.VISIBLE : android.view.View.INVISIBLE);
        });

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build();
                imageAnalysis.setAnalyzer(cameraExecutor, image -> {
                    Bitmap bitmap = image.toBitmap();
                    int rotationDegrees = image.getImageInfo().getRotationDegrees();

                    // Rotate bitmap to portrait orientation
                    // For front camera: rotate then mirror horizontally (selfie mirror effect)
                    Bitmap rotatedBitmap = rotateAndMirrorBitmap(bitmap, rotationDegrees, true);
                    if (bitmap != rotatedBitmap) {
                        bitmap.recycle();
                    }

                    long timestampMs = image.getImageInfo().getTimestamp() / 1000000;
                    if (keypointExtractor != null) {
                        keypointExtractor.detectAsync(rotatedBitmap, timestampMs);
                    }
                    image.close();
                });

                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Use case binding failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    /**
     * Rotate and optionally mirror a bitmap.
     *
     * CameraX returns frames in sensor orientation (landscape for most phones).
     * This method rotates to portrait orientation. For front camera, it also mirrors
     * horizontally to create the selfie-mirror effect that users expect.
     *
     * @param bitmap  Source bitmap (will NOT be recycled by this method)
     * @param degrees  Rotation degrees from sensor (typically 270 for front camera in portrait)
     * @param mirrorH  Whether to mirror horizontally after rotation (true for front camera)
     * @return A new rotated+mirrored bitmap, or the original if rotation is 0 and no mirror
     */
    private static Bitmap rotateAndMirrorBitmap(Bitmap bitmap, int degrees, boolean mirrorH) {
        if (degrees == 0 && !mirrorH) return bitmap;

        Matrix matrix = new Matrix();
        if (degrees != 0) {
            matrix.postRotate(degrees);
        }
        if (mirrorH) {
            matrix.postScale(-1, 1, bitmap.getWidth() / 2f, bitmap.getHeight() / 2f);
        }

        Bitmap result = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        return result;
    }

    @Override
    public void onResults(float[][] keypoints, boolean[] validMask, boolean hasHands) {
        runOnUiThread(() -> {
            if (isSkeletonVisible) {
                skeletonOverlay.setKeypoints(keypoints, validMask);
            }
            if (modelRunner != null) {
                modelRunner.processFrame(keypoints, validMask, hasHands, this);
            }
        });
    }

    @Override
    public void onError(String error) {
        Log.e(TAG, "KeypointExtractor error: " + error);
        runOnUiThread(() -> Toast.makeText(this, error, Toast.LENGTH_SHORT).show());
    }

    @Override
    public void onInferenceResult(String label, float confidence, long inferenceTime) {
        runOnUiThread(() -> {
            tvResult.setText(label.isEmpty() ? "..." : label);
            tvConfidence.setText(String.format("%.2f%%", confidence * 100));
            tvDelay.setText(inferenceTime + " ms");
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        if (keypointExtractor != null) {
            keypointExtractor.close();
        }
    }
}