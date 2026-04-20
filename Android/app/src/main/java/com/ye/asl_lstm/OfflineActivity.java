package com.ye.asl_lstm;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

public class OfflineActivity extends AppCompatActivity implements KeypointExtractor.ExtractorListener, ModelRunner.InferenceListener {
    private static final String TAG = "OfflineActivity";
    private VideoView videoView;
    private View layoutUploadHint;
    private SkeletonOverlayView skeletonOverlay;
    private TextView tvToggleSkeleton;
    private TextView tvConfidence;
    private TextView tvDelay;
    private TextView tvResult;

    private KeypointExtractor keypointExtractor;
    private ModelRunner modelRunner;

    private boolean isSkeletonVisible = true;
    private boolean isExtracting = false;
    private Thread extractionThread;

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    pickVideo();
                } else {
                    Toast.makeText(this, "需要存储权限才能导入视频", Toast.LENGTH_SHORT).show();
                }
            });

    private final ActivityResultLauncher<String> pickVideoLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    playVideo(uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_offline);

        videoView = findViewById(R.id.video_view);
        layoutUploadHint = findViewById(R.id.layout_upload_hint);
        skeletonOverlay = findViewById(R.id.skeleton_overlay);
        tvToggleSkeleton = findViewById(R.id.tv_toggle_skeleton);
        tvConfidence = findViewById(R.id.tv_confidence);
        tvDelay = findViewById(R.id.tv_delay);
        tvResult = findViewById(R.id.tv_result);

        keypointExtractor = new KeypointExtractor(this, this);

        try {
            modelRunner = new ModelRunner(this, "best_model.ptl");
        } catch (Exception e) {
            Toast.makeText(this, "模型加载失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
            Log.e(TAG, "Model load error", e);
        }

        findViewById(R.id.iv_back).setOnClickListener(v -> finish());

        View.OnClickListener uploadListener = v -> pickVideo();
        layoutUploadHint.setOnClickListener(uploadListener);

        tvToggleSkeleton.setOnClickListener(v -> {
            isSkeletonVisible = !isSkeletonVisible;
            tvToggleSkeleton.setText(isSkeletonVisible ? "隐藏骨骼点" : "显示骨骼点");
            skeletonOverlay.setVisibility(isSkeletonVisible ? android.view.View.VISIBLE : android.view.View.INVISIBLE);
        });
    }

    private void pickVideo() {
        pickVideoLauncher.launch("video/*");
    }

    private void playVideo(Uri uri) {
        layoutUploadHint.setVisibility(View.GONE);
        videoView.setVisibility(View.VISIBLE);
        videoView.setVideoURI(uri);
        videoView.setOnPreparedListener(mp -> {
            mp.setLooping(true);
            videoView.start();
            startFrameExtraction(uri);
        });
    }

    private void startFrameExtraction(Uri uri) {
        if (isExtracting) return;
        isExtracting = true;

        extractionThread = new Thread(() -> {
            MediaMetadataRetriever retriever = new MediaMetadataRetriever();
            try {
                retriever.setDataSource(this, uri);
                String timeStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
                long durationMs = Long.parseLong(timeStr);
                // Get video rotation
                String rotationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
                int rotation = 0;
                if (rotationStr != null) {
                    rotation = Integer.parseInt(rotationStr);
                }
                long frameIntervalUs = 33333; // ~30 fps

                for (long timeUs = 0; timeUs < durationMs * 1000; timeUs += frameIntervalUs) {
                    if (!isExtracting) break;

                    Bitmap bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST);
                    if (bitmap != null) {
                        // Rotate frame to match video's orientation
                        Bitmap rotatedBitmap = rotateBitmap(bitmap, rotation);
                        if (bitmap != rotatedBitmap) {
                            bitmap.recycle();
                        }
                        if (keypointExtractor != null) {
                            keypointExtractor.detectAsync(rotatedBitmap, SystemClock.uptimeMillis());
                        }
                    }

                    try {
                        Thread.sleep(33);
                    } catch (InterruptedException e) {
                        break;
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Frame extraction error", e);
            } finally {
                try { retriever.release(); } catch (Exception ignored) {}
                isExtracting = false;
            }
        });
        extractionThread.start();
    }

    /**
     * Rotate a bitmap by the specified degrees.
     * Videos may have rotation metadata (e.g., portrait videos recorded on phone).
     */
    private static Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        if (degrees == 0) return bitmap;

        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
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
            tvResult.setText(label);
            tvConfidence.setText(String.format("%.2f%%", confidence * 100));
            tvDelay.setText(inferenceTime + " ms");
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        isExtracting = false;
        if (extractionThread != null) {
            try {
                extractionThread.join(1000);
            } catch (InterruptedException ignored) {
            }
        }
        if (keypointExtractor != null) {
            keypointExtractor.close();
        }
    }
}