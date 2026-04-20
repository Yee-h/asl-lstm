package com.ye.asl_lstm;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

public class SkeletonOverlayView extends View {
    private float[][] currentKeypoints;
    private boolean[] currentValidMask;

    private final Paint pointPaint;
    private final Paint linePaint;
    private final Paint handPaint;
    private final Paint facePaint;

    // Body 25 connections (OpenPose Body 25 format)
    private static final int[][] BODY_CONNECTIONS = {
            {0, 1}, {1, 2}, {2, 3}, {3, 4},
            {1, 5}, {5, 6}, {6, 7},
            {1, 8}, {8, 9}, {9, 10}, {10, 11},
            {8, 12}, {12, 13}, {13, 14},
            {0, 15}, {0, 16}, {15, 17}, {16, 18}
    };

    // Hand 21 connections
    private static final int[][] HAND_CONNECTIONS = {
            {0, 1}, {1, 2}, {2, 3}, {3, 4},
            {0, 5}, {5, 6}, {6, 7}, {7, 8},
            {0, 9}, {9, 10}, {10, 11}, {11, 12},
            {0, 13}, {13, 14}, {14, 15}, {15, 16},
            {0, 17}, {17, 18}, {18, 19}, {19, 20},
            {5, 9}, {9, 13}, {13, 17}
    };

    public SkeletonOverlayView(Context context) {
        super(context);
        pointPaint = new Paint();
        pointPaint.setColor(Color.RED);
        pointPaint.setStyle(Paint.Style.FILL);
        pointPaint.setStrokeWidth(8f);

        linePaint = new Paint();
        linePaint.setColor(Color.GREEN);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(5f);

        handPaint = new Paint();
        handPaint.setColor(Color.BLUE);
        handPaint.setStyle(Paint.Style.FILL);
        handPaint.setStrokeWidth(6f);

        facePaint = new Paint();
        facePaint.setColor(Color.YELLOW);
        facePaint.setStyle(Paint.Style.STROKE);
        facePaint.setStrokeWidth(3f);
    }

    public SkeletonOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        pointPaint = new Paint();
        pointPaint.setColor(Color.RED);
        pointPaint.setStyle(Paint.Style.FILL);
        pointPaint.setStrokeWidth(8f);

        linePaint = new Paint();
        linePaint.setColor(Color.GREEN);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(5f);

        handPaint = new Paint();
        handPaint.setColor(Color.BLUE);
        handPaint.setStyle(Paint.Style.FILL);
        handPaint.setStrokeWidth(6f);

        facePaint = new Paint();
        facePaint.setColor(Color.YELLOW);
        facePaint.setStyle(Paint.Style.STROKE);
        facePaint.setStrokeWidth(3f);
    }

    public void setKeypoints(float[][] keypoints, boolean[] validMask) {
        this.currentKeypoints = keypoints;
        this.currentValidMask = validMask;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (currentKeypoints == null || currentValidMask == null) return;
        if (currentKeypoints.length < 2) return;

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        float[] xs = currentKeypoints[0];
        float[] ys = currentKeypoints[1];

        // Draw body connections
        for (int[] connection : BODY_CONNECTIONS) {
            int startIdx = connection[0];
            int endIdx = connection[1];
            if (startIdx < currentValidMask.length && endIdx < currentValidMask.length
                    && currentValidMask[startIdx] && currentValidMask[endIdx]) {
                canvas.drawLine(
                        xs[startIdx] * viewWidth, ys[startIdx] * viewHeight,
                        xs[endIdx] * viewWidth, ys[endIdx] * viewHeight,
                        linePaint
                );
            }
        }

        // Draw left hand connections (indices 25-45)
        for (int[] connection : HAND_CONNECTIONS) {
            int startIdx = connection[0] + 25;
            int endIdx = connection[1] + 25;
            if (startIdx < currentValidMask.length && endIdx < currentValidMask.length
                    && currentValidMask[startIdx] && currentValidMask[endIdx]) {
                canvas.drawLine(
                        xs[startIdx] * viewWidth, ys[startIdx] * viewHeight,
                        xs[endIdx] * viewWidth, ys[endIdx] * viewHeight,
                        linePaint
                );
            }
        }

        // Draw right hand connections (indices 46-66)
        for (int[] connection : HAND_CONNECTIONS) {
            int startIdx = connection[0] + 46;
            int endIdx = connection[1] + 46;
            if (startIdx < currentValidMask.length && endIdx < currentValidMask.length
                    && currentValidMask[startIdx] && currentValidMask[endIdx]) {
                canvas.drawLine(
                        xs[startIdx] * viewWidth, ys[startIdx] * viewHeight,
                        xs[endIdx] * viewWidth, ys[endIdx] * viewHeight,
                        linePaint
                );
            }
        }

        // Draw body points (0-24)
        for (int i = 0; i < 25 && i < currentValidMask.length; i++) {
            if (currentValidMask[i]) {
                canvas.drawCircle(xs[i] * viewWidth, ys[i] * viewHeight, 6f, pointPaint);
            }
        }

        // Draw hand points (25-66)
        for (int i = 25; i < 67 && i < currentValidMask.length; i++) {
            if (currentValidMask[i]) {
                canvas.drawCircle(xs[i] * viewWidth, ys[i] * viewHeight, 4f, handPaint);
            }
        }

        // Draw face points (67-134)
        for (int i = 67; i < currentValidMask.length; i++) {
            if (currentValidMask[i]) {
                canvas.drawCircle(xs[i] * viewWidth, ys[i] * viewHeight, 2f, facePaint);
            }
        }
    }
}