package com.ye.asl_lstm;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;

import java.util.List;

public class KeypointExtractor {
    private static final String TAG = "KeypointExtractor";
    public static final int NUM_KEYPOINTS = 135;

    private PoseLandmarker poseLandmarker;
    private HandLandmarker handLandmarker;
    private FaceLandmarker faceLandmarker;
    private final Context context;
    private final ExtractorListener listener;

    private volatile boolean hasHandsLastFrame = false;

    private PoseLandmarkerResult latestPoseResult;
    private HandLandmarkerResult latestHandResult;
    private FaceLandmarkerResult latestFaceResult;
    private volatile boolean frameConsumed = false;

    private FaceLandmarkerResult cachedFaceResult = null;
    private int faceSkipCounter = 0;
    private static final int FACE_DETECT_INTERVAL = 3;

    private final Object resultLock = new Object();

    public interface ExtractorListener {
        void onResults(float[][] keypoints, boolean[] validMask, boolean hasHands);
        void onError(String error);
    }

    public KeypointExtractor(Context context, ExtractorListener listener) {
        this.context = context;
        this.listener = listener;
        setupLandmarkers();
    }

    private void setupLandmarkers() {
        try {
            BaseOptions poseBaseOptions = BaseOptions.builder()
                    .setModelAssetPath("pose_landmarker_heavy.task")
                    .build();
            PoseLandmarker.PoseLandmarkerOptions poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(poseBaseOptions)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumPoses(1)
                    .setMinPoseDetectionConfidence(0.5f)
                    .setMinPosePresenceConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f)
                    .setResultListener(this::onPoseResult)
                    .setErrorListener(this::onPoseError)
                    .build();
            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions);

            BaseOptions handBaseOptions = BaseOptions.builder()
                    .setModelAssetPath("hand_landmarker.task")
                    .build();
            HandLandmarker.HandLandmarkerOptions handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(handBaseOptions)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumHands(2)
                    .setMinHandDetectionConfidence(0.5f)
                    .setMinHandPresenceConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f)
                    .setResultListener(this::onHandResult)
                    .setErrorListener(this::onHandError)
                    .build();
            handLandmarker = HandLandmarker.createFromOptions(context, handOptions);

            BaseOptions faceBaseOptions = BaseOptions.builder()
                    .setModelAssetPath("face_landmarker.task")
                    .build();
            FaceLandmarker.FaceLandmarkerOptions faceOptions = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(faceBaseOptions)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumFaces(1)
                    .setMinFaceDetectionConfidence(0.5f)
                    .setMinFacePresenceConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f)
                    .setResultListener(this::onFaceResult)
                    .setErrorListener(this::onFaceError)
                    .build();
            faceLandmarker = FaceLandmarker.createFromOptions(context, faceOptions);

            Log.i(TAG, "All MediaPipe landmarkers initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize landmarkers: " + e.getMessage(), e);
            if (listener != null) listener.onError("MediaPipe Init Error: " + e.getMessage());
        }
    }

    public void detectAsync(Bitmap bitmap, long timestampMs) {
        if (poseLandmarker == null || handLandmarker == null || faceLandmarker == null) return;

        bitmap = downsampleIfNeeded(bitmap);
        MPImage mpImage = new BitmapImageBuilder(bitmap).build();
        synchronized (resultLock) {
            frameConsumed = false;
            latestPoseResult = null;
            latestHandResult = null;
            latestFaceResult = null;
        }

        handLandmarker.detectAsync(mpImage, timestampMs);

        if (hasHandsLastFrame) {
            poseLandmarker.detectAsync(mpImage, timestampMs);

            faceSkipCounter++;
            if (faceSkipCounter >= FACE_DETECT_INTERVAL) {
                faceLandmarker.detectAsync(mpImage, timestampMs);
                faceSkipCounter = 0;
            }
        } else {
            faceSkipCounter = FACE_DETECT_INTERVAL;
        }
    }

    private static final int DOWNSAMPLE_MAX_DIM = 480;

    private Bitmap downsampleIfNeeded(Bitmap bitmap) {
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        if (w <= DOWNSAMPLE_MAX_DIM && h <= DOWNSAMPLE_MAX_DIM) return bitmap;
        float scale = (float) DOWNSAMPLE_MAX_DIM / Math.max(w, h);
        int newW = Math.round(w * scale);
        int newH = Math.round(h * scale);
        return Bitmap.createScaledBitmap(bitmap, newW, newH, true);
    }

    private void onPoseResult(PoseLandmarkerResult result, MPImage image) {
        synchronized (resultLock) {
            latestPoseResult = result;
        }
        tryEmit();
    }

    private void onHandResult(HandLandmarkerResult result, MPImage image) {
        synchronized (resultLock) {
            latestHandResult = result;
        }
        boolean hasHands = result.landmarks() != null && !result.landmarks().isEmpty();
        hasHandsLastFrame = hasHands;
        tryEmit();
    }

    private void onFaceResult(FaceLandmarkerResult result, MPImage image) {
        synchronized (resultLock) {
            latestFaceResult = result;
            if (result.faceLandmarks() != null && !result.faceLandmarks().isEmpty()) {
                cachedFaceResult = result;
            }
        }
        tryEmit();
    }

    private void onPoseError(RuntimeException e) {
        Log.e(TAG, "Pose error: " + e.getMessage());
    }

    private void onHandError(RuntimeException e) {
        Log.e(TAG, "Hand error: " + e.getMessage());
    }

    private void onFaceError(RuntimeException e) {
        Log.e(TAG, "Face error: " + e.getMessage());
    }

    private void tryEmit() {
        PoseLandmarkerResult poseResult;
        HandLandmarkerResult handResult;
        FaceLandmarkerResult faceResult;

        synchronized (resultLock) {
            if (frameConsumed) return;

            handResult = latestHandResult;
            poseResult = latestPoseResult;
            faceResult = latestFaceResult;

            if (handResult == null) return;

            boolean hasHands = handResult.landmarks() != null && !handResult.landmarks().isEmpty();

            if (!hasHands) {
                frameConsumed = true;
                if (listener != null) {
                    float[][] kp = new float[2][NUM_KEYPOINTS];
                    boolean[] mask = new boolean[NUM_KEYPOINTS];
                    listener.onResults(kp, mask, false);
                }
                return;
            }

            if (poseResult == null) return;

            if (faceResult == null && cachedFaceResult != null && hasHands) {
                faceResult = cachedFaceResult;
            }

            frameConsumed = true;
        }

        mapTo135Keypoints(poseResult, handResult, faceResult);
    }

    private void mapTo135Keypoints(
            PoseLandmarkerResult poseResult,
            HandLandmarkerResult handResult,
            FaceLandmarkerResult faceResult) {

        float[][] keypoints = new float[2][NUM_KEYPOINTS];
        boolean[] validMask = new boolean[NUM_KEYPOINTS];

        // 1. Body 25 points mapping
        int[] bodyMapping = {
                0,   // 0: Nose
                -1,  // 1: Neck = (MP11 + MP12) / 2
                12,  // 2: RShoulder
                14,  // 3: RElbow
                16,  // 4: RWrist
                11,  // 5: LShoulder
                13,  // 6: LElbow
                15,  // 7: LWrist
                -1,  // 8: MidHip = (MP23 + MP24) / 2
                24,  // 9: RHip
                26,  // 10: RKnee
                28,  // 11: RAnkle
                23,  // 12: LHip
                25,  // 13: LKnee
                27,  // 14: LAnkle
                5,   // 15: REye
                2,   // 16: LEye
                8,   // 17: REar
                7,   // 18: LEar
                32,  // 19: LBigToe
                31,  // 20: LSmallToe
                29,  // 21: LHeel
                31,  // 22: RBigToe
                32,  // 23: RSmallToe
                30   // 24: RHeel
        };

        if (poseResult.landmarks() != null && !poseResult.landmarks().isEmpty()) {
            List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> pose =
                    poseResult.landmarks().get(0);

            for (int i = 0; i < bodyMapping.length; i++) {
                int mpIdx = bodyMapping[i];
                if (mpIdx >= 0 && mpIdx < pose.size()) {
                    keypoints[0][i] = pose.get(mpIdx).x();
                    keypoints[1][i] = pose.get(mpIdx).y();
                    validMask[i] = true;
                } else if (i == 1 && pose.size() > 12) {
                    // Neck = (LeftShoulder + RightShoulder) / 2
                    keypoints[0][i] = (pose.get(11).x() + pose.get(12).x()) / 2f;
                    keypoints[1][i] = (pose.get(11).y() + pose.get(12).y()) / 2f;
                    validMask[i] = true;
                } else if (i == 8 && pose.size() > 24) {
                    // MidHip = (LeftHip + RightHip) / 2
                    keypoints[0][i] = (pose.get(23).x() + pose.get(24).x()) / 2f;
                    keypoints[1][i] = (pose.get(23).y() + pose.get(24).y()) / 2f;
                    validMask[i] = true;
                }
            }
        }

        // 2. Hand 42 points mapping
        if (handResult.landmarks() != null && !handResult.landmarks().isEmpty()) {
            for (int handIdx = 0; handIdx < handResult.landmarks().size() && handIdx < handResult.handedness().size(); handIdx++) {
                List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> handLms =
                        handResult.landmarks().get(handIdx);
                if (handResult.handedness().get(handIdx).isEmpty()) continue;
                String handLabel = handResult.handedness().get(handIdx).get(0).categoryName().toLowerCase();

                // MediaPipe 'left' (image left) -> human right hand -> OpenPose Right Hand (46-66)
                // MediaPipe 'right' (image right) -> human left hand -> OpenPose Left Hand (25-45)
                int offset;
                if (handLabel.equals("left")) {
                    offset = 46; // OpenPose Right Hand
                } else {
                    offset = 25; // OpenPose Left Hand
                }

                for (int i = 0; i < handLms.size() && i < 21; i++) {
                    keypoints[0][offset + i] = handLms.get(i).x();
                    keypoints[1][offset + i] = handLms.get(i).y();
                    validMask[offset + i] = true;
                }
            }
        }

        // 3. Face 68 points mapping
        int[] face68Indices = {
                // Face outline (17)
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
                // Left eyebrow (5)
                70, 63, 105, 66, 107,
                // Right eyebrow (5)
                336, 296, 334, 293, 300,
                // Nose bridge (4)
                168, 6, 197, 195,
                // Nose bottom (5)
                5, 4, 1, 19, 94,
                // Left eye (6)
                33, 160, 158, 133, 153, 144,
                // Right eye (6)
                362, 385, 387, 263, 373, 380,
                // Outer lip (12)
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308,
                // Inner lip (8)
                78, 191, 80, 81, 82, 13, 312, 311
        };

        if (faceResult != null && faceResult.faceLandmarks() != null && !faceResult.faceLandmarks().isEmpty()) {
            List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> faceLms =
                    faceResult.faceLandmarks().get(0);

            for (int i = 0; i < face68Indices.length && i < 68; i++) {
                int mpIdx = face68Indices[i];
                if (mpIdx < faceLms.size()) {
                    keypoints[0][67 + i] = faceLms.get(mpIdx).x();
                    keypoints[1][67 + i] = faceLms.get(mpIdx).y();
                    validMask[67 + i] = true;
                }
            }
        }

        boolean hasHands = handResult.landmarks() != null && !handResult.landmarks().isEmpty();
        if (listener != null) {
            listener.onResults(keypoints, validMask, hasHands);
        }
    }

    public void close() {
        if (poseLandmarker != null) {
            poseLandmarker.close();
            poseLandmarker = null;
        }
        if (handLandmarker != null) {
            handLandmarker.close();
            handLandmarker = null;
        }
        if (faceLandmarker != null) {
            faceLandmarker.close();
            faceLandmarker = null;
        }
    }
}