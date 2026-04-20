package com.ye.asl_lstm;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class ModelRunner {
    private static final String TAG = "ModelRunner";

    public static final String[] ASL_LABELS = new String[]{
            "book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy", "cousin",
            "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all", "black",
            "cool", "finish", "hot", "like", "many", "mother", "now", "orange", "table", "thanksgiving",
            "what", "woman", "bed", "blue", "bowling", "can", "dog", "family", "fish", "graduate",
            "hat", "hearing", "kiss", "language", "later", "man", "shirt", "study", "tall", "white",
            "wrong", "accident", "apple", "bird", "change", "color", "corn", "cow", "dance", "dark",
            "doctor", "eat", "enjoy", "forget", "give", "last", "meet", "pink", "pizza", "play",
            "school", "secretary", "short", "time", "want", "work", "africa", "basketball", "birthday", "brown",
            "but", "cheat", "city", "cook", "decide", "full", "how", "jacket", "letter", "medicine",
            "need", "paint", "paper", "pull", "purple", "right", "same", "son", "tell", "thursday"
    };

    private static final int MAX_FRAMES = 90;
    private static final int NUM_KEYPOINTS = 135;
    private static final int FEATURE_DIM = 540; // 135 * 4 (x, y, dx, dy)

    private Module module;
    private final PreprocessPipeline pipeline;

    private int noKeypointFrameCount = 0;
    private static final int NO_KEYPOINT_THRESHOLD = 15;
    private static final float CONFIDENCE_THRESHOLD = 0.15f;

    private InferenceListener pendingListener = null;

    public interface InferenceListener {
        void onInferenceResult(String label, float confidence, long inferenceTime);
        void onError(String error);
    }

    public ModelRunner(Context context, String modelName) throws Exception {
        File file = new File(context.getFilesDir(), modelName);
        if (!file.exists()) {
            try (InputStream is = context.getAssets().open(modelName);
                 OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4096];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
            }
        }
        module = LiteModuleLoader.load(file.getAbsolutePath());
        pipeline = new PreprocessPipeline();
    }

    public void processFrame(float[][] keypoints, boolean[] validMask, boolean hasHands, InferenceListener listener) {
        if (module == null) return;

        if (!hasHands) {
            noKeypointFrameCount++;
            // Try inference if we have enough frames and hands disappeared
            if (pipeline.shouldInfer(noKeypointFrameCount)) {
                runInference(listener);
                pipeline.clearBuffer();
                noKeypointFrameCount = 0;
            }
            return;
        }

        noKeypointFrameCount = 0;
        pipeline.addFrame(keypoints, validMask);
        pendingListener = listener;

        if (pipeline.getBufferSize() >= MAX_FRAMES) {
            runInference(listener);
            pipeline.clearBuffer();
        }
    }

    public void forceInference(InferenceListener listener) {
        if (module == null || pipeline.getBufferSize() < 5) return;
        runInference(listener);
        pipeline.clearBuffer();
    }

    private void runInference(InferenceListener listener) {
        if (module == null) return;

        long startTime = SystemClock.uptimeMillis();

        try {
            PreprocessPipeline.ProcessedResult result = pipeline.process();
            if (result == null) return;

            float[] xData = result.data;
            int currentLength = result.validLength;

            long[] xShape = new long[]{1, MAX_FRAMES, FEATURE_DIM};
            Tensor xTensor = Tensor.fromBlob(xData, xShape);

            long[] lenData = new long[]{currentLength};
            long[] lenShape = new long[]{1};
            Tensor lenTensor = Tensor.fromBlob(lenData, lenShape);

            IValue out = module.forward(IValue.from(xTensor), IValue.from(lenTensor));
            float[] logits = out.toTensor().getDataAsFloatArray();

            // Softmax
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (float logit : logits) maxLogit = Math.max(maxLogit, logit);

            float sumExp = 0;
            float[] probs = new float[logits.length];
            for (int i = 0; i < logits.length; i++) {
                probs[i] = (float) Math.exp(logits[i] - maxLogit);
                sumExp += probs[i];
            }

            int maxIndex = 0;
            float maxProb = 0;
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sumExp;
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }

            long inferenceTime = SystemClock.uptimeMillis() - startTime;
            String label = maxProb >= CONFIDENCE_THRESHOLD ? ASL_LABELS[maxIndex] : "";

            if (listener != null) {
                listener.onInferenceResult(label, maxProb, inferenceTime);
            }

        } catch (Exception e) {
            Log.e(TAG, "Inference error", e);
            if (listener != null) listener.onError(e.getMessage());
        }
    }

    public void reset() {
        pipeline.resetAll();
        noKeypointFrameCount = 0;
    }
}