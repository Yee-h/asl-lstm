package com.ye.asl_lstm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PreprocessPipeline {
    private static final String TAG = "PreprocessPipeline";

    public static final int NUM_KEYPOINTS = 135;
    public static final int MAX_FRAMES = 90;

    private static final float SMOOTH_ALPHA = 0.35f;
    private static final int MAX_INTERP_GAP = 8;
    private static final float NORMALIZE_EPS = 1e-6f;

    // Z-Score standardization constants
    private static final float[] MEAN = {-0.02456033f, -0.11962947f, -0.00081918f, 0.00250032f};
    private static final float[] STD = {0.24632293f, 0.69178674f, 0.03799942f, 0.06176139f};
    private static final float ZSCORE_EPS = 1e-6f;

    // Sliding window for scale normalization (online mode)
    private static final int SCALE_WINDOW_SIZE = 300;
    private static final int SCALE_MIN_FRAMES = 30;

    private final List<float[]> shoulderDists = new ArrayList<>();
    private final List<float[]> torsoDists = new ArrayList<>();

    private float[] lastFramePoints = null;
    private boolean emaInitialized = false;
    private float[][] emaState = null;
    private int frameCount = 0;

    // Buffer for per-frame data before processing
    private final List<float[]> frameBuffer = new ArrayList<>();
    private final List<boolean[]> frameMasks = new ArrayList<>();

    public PreprocessPipeline() {
    }

    public void addFrame(float[][] keypoints, boolean[] validMask) {
        // Store as (135, 2) internally
        float[] frame = new float[NUM_KEYPOINTS * 2];
        boolean[] mask = new boolean[NUM_KEYPOINTS];

        for (int i = 0; i < NUM_KEYPOINTS; i++) {
            frame[i] = keypoints[0][i];          // x
            frame[NUM_KEYPOINTS + i] = keypoints[1][i]; // y
            mask[i] = validMask[i];
        }

        frameBuffer.add(frame);
        frameMasks.add(mask);
    }

    public int getBufferSize() {
        return frameBuffer.size();
    }

    public boolean shouldInfer(int noKeypointFrames) {
        if (frameBuffer.size() >= MAX_FRAMES) return true;
        if (frameBuffer.size() >= 15 && noKeypointFrames >= 15) return true;
        return false;
    }

    public void clearBuffer() {
        frameBuffer.clear();
        frameMasks.clear();
        lastFramePoints = null;
        emaInitialized = false;
        emaState = null;
        shoulderDists.clear();
        torsoDists.clear();
        frameCount = 0;
    }

    public ProcessedResult process() {
        if (frameBuffer.isEmpty()) return null;

        int T = Math.min(frameBuffer.size(), MAX_FRAMES);

        // Convert to float[T][135][2] format
        float[][][] data = new float[T][NUM_KEYPOINTS][2];
        boolean[][] masks = new boolean[T][NUM_KEYPOINTS];

        for (int t = 0; t < T; t++) {
            float[] frame = frameBuffer.get(t);
            boolean[] mask = frameMasks.get(t);
            for (int v = 0; v < NUM_KEYPOINTS; v++) {
                data[t][v][0] = frame[v];                // x
                data[t][v][1] = frame[NUM_KEYPOINTS + v]; // y
                masks[t][v] = mask[v];
            }
        }

        // 1. Interpolate short gaps
        interpolateMissingShortGaps(data, masks, T);

        // 2. EMA smoothing
        smoothXY(data, masks, T);

        // 3. Shoulder axis alignment
        alignShoulderAxis(data, masks, T);

        // 4. Scale normalization
        normalizeWithMask(data, masks, T);

        // 5. Add velocity features (dx, dy)
        float[][][] featureData = addVelocityFeatures(data, T);

        // 6. Z-Score standardization
        applyStandardization(featureData, masks, T);

        // 7. Pad/truncate to MAX_FRAMES, zero out mask=0 points
        return padOrTruncate(featureData, masks, T);
    }

    // 1. Interpolate short gaps (gap <= MAX_INTERP_GAP)
    private void interpolateMissingShortGaps(float[][][] data, boolean[][] masks, int T) {
        for (int v = 0; v < NUM_KEYPOINTS; v++) {
            int t = 0;
            while (t < T) {
                if (masks[t][v]) {
                    t++;
                    continue;
                }

                int start = t;
                while (t < T && !masks[t][v]) t++;
                int end = t - 1;
                int gapLen = end - start + 1;

                if (gapLen > MAX_INTERP_GAP) continue;
                int left = start - 1;
                int right = t;

                if (left < 0 || right >= T) continue;
                if (!masks[left][v] || !masks[right][v]) continue;

                for (int k = 0; k < gapLen; k++) {
                    float ratio = (float) (k + 1) / (gapLen + 1);
                    for (int c = 0; c < 2; c++) {
                        float leftVal = data[left][v][c];
                        float rightVal = data[right][v][c];
                        data[start + k][v][c] = leftVal + (rightVal - leftVal) * ratio;
                    }
                    masks[start + k][v] = true;
                }
            }
        }
    }

    // 2. EMA smoothing (alpha=0.35, only on valid points)
    private void smoothXY(float[][][] data, boolean[][] masks, int T) {
        if (!emaInitialized) {
            emaState = new float[NUM_KEYPOINTS][2];
            emaInitialized = true;
        }

        for (int v = 0; v < NUM_KEYPOINTS; v++) {
            boolean foundFirst = false;
            for (int t = 0; t < T; t++) {
                if (!masks[t][v]) continue;
                if (!foundFirst) {
                    emaState[v][0] = data[t][v][0];
                    emaState[v][1] = data[t][v][1];
                    foundFirst = true;
                    continue;
                }
                emaState[v][0] = SMOOTH_ALPHA * data[t][v][0] + (1 - SMOOTH_ALPHA) * emaState[v][0];
                emaState[v][1] = SMOOTH_ALPHA * data[t][v][1] + (1 - SMOOTH_ALPHA) * emaState[v][1];
                data[t][v][0] = emaState[v][0];
                data[t][v][1] = emaState[v][1];
            }
        }
    }

    // 3. Shoulder axis alignment (per frame)
    private void alignShoulderAxis(float[][][] data, boolean[][] masks, int T) {
        int LEFT_SHOULDER = 5;
        int RIGHT_SHOULDER = 2;

        for (int t = 0; t < T; t++) {
            if (!masks[t][LEFT_SHOULDER] || !masks[t][RIGHT_SHOULDER]) continue;

            float lsX = data[t][LEFT_SHOULDER][0];
            float lsY = data[t][LEFT_SHOULDER][1];
            float rsX = data[t][RIGHT_SHOULDER][0];
            float rsY = data[t][RIGHT_SHOULDER][1];

            float vecX = lsX - rsX;
            float vecY = lsY - rsY;
            float norm = (float) Math.sqrt(vecX * vecX + vecY * vecY);
            if (norm <= NORMALIZE_EPS) continue;

            double angle = Math.atan2(vecY, vecX);
            double cosA = Math.cos(-angle);
            double sinA = Math.sin(-angle);

            float rootX = (lsX + rsX) / 2f;
            float rootY = (lsY + rsY) / 2f;

            for (int v = 0; v < NUM_KEYPOINTS; v++) {
                if (!masks[t][v]) continue;
                float px = data[t][v][0] - rootX;
                float py = data[t][v][1] - rootY;
                data[t][v][0] = (float) (cosA * px - sinA * py) + rootX;
                data[t][v][1] = (float) (sinA * px + cosA * py) + rootY;
            }
        }
    }

    // 4. Scale normalization (online mode with sliding window)
    private void normalizeWithMask(float[][][] data, boolean[][] masks, int T) {
        int LEFT_SHOULDER = 5;
        int RIGHT_SHOULDER = 2;
        int MID_HIP = 8;

        float[][] roots = new float[T][2];
        List<Float> shoulderScales = new ArrayList<>();
        List<Float> torsoScales = new ArrayList<>();

        for (int t = 0; t < T; t++) {
            boolean hasLeft = masks[t][LEFT_SHOULDER];
            boolean hasRight = masks[t][RIGHT_SHOULDER];

            if (hasLeft && hasRight) {
                float lsX = data[t][LEFT_SHOULDER][0];
                float lsY = data[t][LEFT_SHOULDER][1];
                float rsX = data[t][RIGHT_SHOULDER][0];
                float rsY = data[t][RIGHT_SHOULDER][1];
                roots[t][0] = (lsX + rsX) / 2f;
                roots[t][1] = (lsY + rsY) / 2f;
                float shoulderDist = (float) Math.sqrt((lsX - rsX) * (lsX - rsX) + (lsY - rsY) * (lsY - rsY));
                if (shoulderDist > NORMALIZE_EPS) {
                    shoulderScales.add(shoulderDist);
                }
            } else if (t > 0) {
                roots[t][0] = roots[t - 1][0];
                roots[t][1] = roots[t - 1][1];
            }

            if (masks[t][MID_HIP]) {
                float torsoDist = (float) Math.sqrt(
                        (roots[t][0] - data[t][MID_HIP][0]) * (roots[t][0] - data[t][MID_HIP][0]) +
                        (roots[t][1] - data[t][MID_HIP][1]) * (roots[t][1] - data[t][MID_HIP][1]));
                if (torsoDist > NORMALIZE_EPS) {
                    torsoScales.add(torsoDist);
                }
            }
        }

        // Add to sliding window
        for (Float s : shoulderScales) shoulderDists.add(new float[]{s});
        for (Float s : torsoScales) torsoDists.add(new float[]{s});
        while (shoulderDists.size() > SCALE_WINDOW_SIZE) shoulderDists.remove(0);
        while (torsoDists.size() > SCALE_WINDOW_SIZE) torsoDists.remove(0);

        // Compute reference scale
        float videoScale;
        if (shoulderDists.size() < SCALE_MIN_FRAMES) {
            videoScale = 1.0f;
        } else {
            float shoulderMedian = median(shoulderDists);
            float torsoMedian = median(torsoDists);
            videoScale = 0.7f * shoulderMedian + 0.3f * torsoMedian;
        }

        if (videoScale <= NORMALIZE_EPS || Float.isNaN(videoScale)) {
            videoScale = 1.0f;
        }

        // Normalize
        for (int t = 0; t < T; t++) {
            for (int v = 0; v < NUM_KEYPOINTS; v++) {
                if (!masks[t][v]) {
                    data[t][v][0] = 0;
                    data[t][v][1] = 0;
                    continue;
                }
                data[t][v][0] = (data[t][v][0] - roots[t][0]) / videoScale;
                data[t][v][1] = (data[t][v][1] - roots[t][1]) / videoScale;
            }
        }
    }

    private float median(List<float[]> values) {
        if (values.isEmpty()) return 1.0f;
        List<Float> sorted = new ArrayList<>();
        for (float[] v : values) sorted.add(v[0]);
        Collections.sort(sorted);
        int n = sorted.size();
        if (n % 2 == 0) {
            return (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2f;
        }
        return sorted.get(n / 2);
    }

    // 5. Add velocity features (dx, dy)
    private float[][][] addVelocityFeatures(float[][][] data, int T) {
        // Output shape: [T][135][4] (x, y, dx, dy)
        float[][][] result = new float[T][NUM_KEYPOINTS][4];

        for (int t = 0; t < T; t++) {
            for (int v = 0; v < NUM_KEYPOINTS; v++) {
                result[t][v][0] = data[t][v][0]; // x
                result[t][v][1] = data[t][v][1]; // y

                if (t > 0) {
                    result[t][v][2] = data[t][v][0] - data[t - 1][v][0]; // dx
                    result[t][v][3] = data[t][v][1] - data[t - 1][v][1]; // dy
                }
                // dx, dy at t=0 remain 0
            }
        }

        return result;
    }

    // 6. Z-Score standardization
    private void applyStandardization(float[][][] featureData, boolean[][] masks, int T) {
        // featureData shape: [T][135][4]
        // MEAN/STD indexed by channel: [x, y, dx, dy]
        for (int t = 0; t < T; t++) {
            for (int v = 0; v < NUM_KEYPOINTS; v++) {
                boolean valid = masks[t][v];
                for (int c = 0; c < 4; c++) {
                    if (valid) {
                        featureData[t][v][c] = (featureData[t][v][c] - MEAN[c]) / (STD[c] + ZSCORE_EPS);
                    } else {
                        featureData[t][v][c] = 0f;
                    }
                }
            }
        }
    }

    // 7. Pad or truncate to MAX_FRAMES, flatten to model input format
    // Model expects (T, C*V) where C=4, V=135, channels grouped:
    //   [x0,x1,...,x134, y0,y1,...,y134, dx0,...,dx134, dy0,...,dy134]
    private ProcessedResult padOrTruncate(float[][][] featureData, boolean[][] masks, int validLen) {
        int seqLen = Math.min(featureData.length, MAX_FRAMES);
        int effLen = Math.min(validLen, MAX_FRAMES);

        float[] flatData = new float[MAX_FRAMES * 540];

        for (int t = 0; t < MAX_FRAMES; t++) {
            int tOffset = t * 540;
            if (t >= seqLen) continue;

            for (int v = 0; v < NUM_KEYPOINTS; v++) {
                boolean valid = (t < effLen) && masks[t][v];
                for (int c = 0; c < 4; c++) {
                    // Channel-grouped: c*135+v, matching Python (T, C, V).flatten()
                    int idx = tOffset + c * NUM_KEYPOINTS + v;
                    flatData[idx] = valid ? featureData[t][v][c] : 0f;
                }
            }
        }

        return new ProcessedResult(flatData, effLen);
    }

    public static class ProcessedResult {
        public final float[] data; // [MAX_FRAMES * 540] flat array
        public final int validLength;

        public ProcessedResult(float[] data, int validLength) {
            this.data = data;
            this.validLength = validLength;
        }
    }
}