package com.zygisk_enc.RecorderX;

import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.Matrix;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Surface;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

// Composites the mirror into the encoder surface so the framing is ours, not the system's
class CaptureRenderer {
    private static final String TAG = "RecorderX_Renderer";
    private static final int EGL_RECORDABLE_ANDROID = 0x3142;
    private static final long GEOMETRY_SETTLE_NS = 200_000_000L;
    private static final long HEARTBEAT_MS = 250;

    private static final String VERTEX_SHADER =
        "uniform mat4 uMvp;\n" +
        "uniform mat4 uTexMatrix;\n" +
        "attribute vec4 aPosition;\n" +
        "attribute vec4 aTextureCoord;\n" +
        "varying vec2 vTextureCoord;\n" +
        "void main() {\n" +
        "    gl_Position = uMvp * aPosition;\n" +
        "    vTextureCoord = (uTexMatrix * aTextureCoord).xy;\n" +
        "}\n";

    private static final String FRAGMENT_SHADER =
        "#extension GL_OES_EGL_image_external : require\n" +
        "precision mediump float;\n" +
        "varying vec2 vTextureCoord;\n" +
        "uniform samplerExternalOES sTexture;\n" +
        "void main() {\n" +
        "    gl_FragColor = texture2D(sTexture, vTextureCoord);\n" +
        "}\n";

    private static final float[] QUAD = {
        -1f, -1f, 0f, 0f,
         1f, -1f, 1f, 0f,
        -1f,  1f, 0f, 1f,
         1f,  1f, 1f, 1f,
    };

    private final int outputWidth;
    private final int outputHeight;

    private final HandlerThread thread;
    private final Handler handler;

    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;

    private int program;
    private int aPositionLoc;
    private int aTextureCoordLoc;
    private int uMvpLoc;
    private int uTexMatrixLoc;

    private int textureId;
    private SurfaceTexture surfaceTexture;
    private Surface inputSurface;
    private FloatBuffer quad;

    private final float[] mvp = new float[16];
    private final float[] texMatrix = new float[16];

    private int sourceWidth;
    private int sourceHeight;
    private final boolean allowRotation;
    private int displayRotation;
    private long holdUntilNs;
    private final long minFrameIntervalNs;
    private long lastPresentedNs;
    private long nextPresentNs;
    private volatile boolean released;
    private volatile boolean paused;
    private final Runnable heartbeat = this::drawHeartbeat;

    CaptureRenderer(Surface encoderSurface, int outputWidth, int outputHeight,
                    int sourceWidth, int sourceHeight, boolean allowRotation,
                    int displayRotation, int targetFps) throws RuntimeException {
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        this.sourceWidth = sourceWidth;
        this.sourceHeight = sourceHeight;
        this.allowRotation = allowRotation;
        this.displayRotation = displayRotation;
        this.minFrameIntervalNs = targetFps > 0 ? 1_000_000_000L / targetFps : 0L;

        thread = new HandlerThread("CaptureRenderer");
        thread.start();
        handler = new Handler(thread.getLooper());

        final Object lock = new Object();
        final RuntimeException[] failure = new RuntimeException[1];
        final boolean[] done = new boolean[1];

        handler.post(() -> {
            try {
                setupEgl(encoderSurface);
                setupGl();
            } catch (RuntimeException e) {
                failure[0] = e;
            }
            synchronized (lock) {
                done[0] = true;
                lock.notifyAll();
            }
        });

        synchronized (lock) {
            while (!done[0]) {
                try { lock.wait(5000); } catch (InterruptedException ignored) { break; }
            }
        }
        if (failure[0] != null) {
            release();
            throw failure[0];
        }
    }

    Surface getInputSurface() {
        return inputSurface;
    }

    // The display changed shape, so re-fit. Called from the service's configuration callback.
    void setSourceSize(int width, int height, int rotation) {
        if (released || width <= 0 || height <= 0) return;
        handler.post(() -> {
            if (released || surfaceTexture == null) return;
            sourceWidth = width;
            sourceHeight = height;
            displayRotation = rotation;
            holdUntilNs = System.nanoTime() + GEOMETRY_SETTLE_NS;
            surfaceTexture.setDefaultBufferSize(width, height);
            Log.i(TAG, "Source size now " + width + "x" + height + " rotation " + rotation);
        });
    }

    void release() {
        if (released) return;
        released = true;
        handler.removeCallbacks(heartbeat);
        handler.post(() -> {
            if (program != 0) {
                GLES20.glDeleteProgram(program);
                program = 0;
            }
            if (textureId != 0) {
                GLES20.glDeleteTextures(1, new int[]{textureId}, 0);
                textureId = 0;
            }
            if (surfaceTexture != null) {
                surfaceTexture.setOnFrameAvailableListener(null);
                surfaceTexture.release();
                surfaceTexture = null;
            }
            if (inputSurface != null) {
                inputSurface.release();
                inputSurface = null;
            }
            if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
                EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
                if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(eglDisplay, eglSurface);
                if (eglContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(eglDisplay, eglContext);
                EGL14.eglTerminate(eglDisplay);
            }
            eglDisplay = EGL14.EGL_NO_DISPLAY;
            eglContext = EGL14.EGL_NO_CONTEXT;
            eglSurface = EGL14.EGL_NO_SURFACE;
            thread.quitSafely();
        });
    }

    private void setupEgl(Surface encoderSurface) {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) throw new RuntimeException("eglGetDisplay failed");

        int[] version = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw new RuntimeException("eglInitialize failed");
        }

        int[] configAttrs = {
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL_RECORDABLE_ANDROID, 1,
            EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, configAttrs, 0, configs, 0, 1, numConfigs, 0)
                || numConfigs[0] <= 0) {
            throw new RuntimeException("eglChooseConfig failed");
        }

        int[] contextAttrs = { EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE };
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttrs, 0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) throw new RuntimeException("eglCreateContext failed");

        int[] surfaceAttrs = { EGL14.EGL_NONE };
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], encoderSurface, surfaceAttrs, 0);
        if (eglSurface == EGL14.EGL_NO_SURFACE) throw new RuntimeException("eglCreateWindowSurface failed");

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw new RuntimeException("eglMakeCurrent failed");
        }
    }

    private void setupGl() {
        program = buildProgram();
        aPositionLoc = GLES20.glGetAttribLocation(program, "aPosition");
        aTextureCoordLoc = GLES20.glGetAttribLocation(program, "aTextureCoord");
        uMvpLoc = GLES20.glGetUniformLocation(program, "uMvp");
        uTexMatrixLoc = GLES20.glGetUniformLocation(program, "uTexMatrix");

        quad = ByteBuffer.allocateDirect(QUAD.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        quad.put(QUAD).position(0);

        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        surfaceTexture = new SurfaceTexture(textureId);
        surfaceTexture.setDefaultBufferSize(sourceWidth, sourceHeight);
        surfaceTexture.setOnFrameAvailableListener(st -> handler.post(this::drawFrame));
        inputSurface = new Surface(surfaceTexture);
    }

    private int buildProgram() {
        int vertex = compile(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
        int fragment = compile(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
        int id = GLES20.glCreateProgram();
        GLES20.glAttachShader(id, vertex);
        GLES20.glAttachShader(id, fragment);
        GLES20.glLinkProgram(id);
        int[] linked = new int[1];
        GLES20.glGetProgramiv(id, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] != GLES20.GL_TRUE) {
            String log = GLES20.glGetProgramInfoLog(id);
            GLES20.glDeleteProgram(id);
            throw new RuntimeException("Program link failed: " + log);
        }
        GLES20.glDeleteShader(vertex);
        GLES20.glDeleteShader(fragment);
        return id;
    }

    private int compile(int type, String source) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] != GLES20.GL_TRUE) {
            String log = GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader);
            throw new RuntimeException("Shader compile failed: " + log);
        }
        return shader;
    }

    private void drawFrame() {
        if (released || surfaceTexture == null || eglDisplay == EGL14.EGL_NO_DISPLAY) return;
        try {
            surfaceTexture.updateTexImage();

            // Drain but do not present while the geometry swaps, so the encoder repeats the last
            if (System.nanoTime() < holdUntilNs) return;

            // The mirror runs at the panel's refresh rate, so drop frames to hit the chosen fps
            long frameNs = surfaceTexture.getTimestamp();
            if (minFrameIntervalNs > 0) {
                if (nextPresentNs == 0) {
                    nextPresentNs = frameNs;
                } else if (frameNs + minFrameIntervalNs / 8 < nextPresentNs) {
                    return;
                }
                nextPresentNs += minFrameIntervalNs;
                if (nextPresentNs <= frameNs) nextPresentNs = frameNs + minFrameIntervalNs;
            }
            if (frameNs <= lastPresentedNs) frameNs = lastPresentedNs + 1;

            surfaceTexture.getTransformMatrix(texMatrix);
            renderAt(frameNs);
        } catch (Exception e) {
            Log.e(TAG, "Frame draw failed", e);
        }
    }

    // A still screen produces no frames at all, so keep the timeline fed while nothing moves
    private void drawHeartbeat() {
        if (released || paused || surfaceTexture == null || eglDisplay == EGL14.EGL_NO_DISPLAY) return;
        if (lastPresentedNs == 0) return;
        if (System.nanoTime() < holdUntilNs) {
            armHeartbeat();
            return;
        }
        try {
            renderAt(Math.max(System.nanoTime(), lastPresentedNs + 1));
        } catch (Exception e) {
            Log.e(TAG, "Heartbeat draw failed", e);
        }
    }

    private void armHeartbeat() {
        handler.removeCallbacks(heartbeat);
        handler.postDelayed(heartbeat, HEARTBEAT_MS);
    }

    private void renderAt(long frameNs) {
        try {
            lastPresentedNs = frameNs;
            computeMvp();

            GLES20.glViewport(0, 0, outputWidth, outputHeight);
            GLES20.glClearColor(0f, 0f, 0f, 1f);
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

            GLES20.glUseProgram(program);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);

            quad.position(0);
            GLES20.glVertexAttribPointer(aPositionLoc, 2, GLES20.GL_FLOAT, false, 16, quad);
            GLES20.glEnableVertexAttribArray(aPositionLoc);
            quad.position(2);
            GLES20.glVertexAttribPointer(aTextureCoordLoc, 2, GLES20.GL_FLOAT, false, 16, quad);
            GLES20.glEnableVertexAttribArray(aTextureCoordLoc);

            GLES20.glUniformMatrix4fv(uMvpLoc, 1, false, mvp, 0);
            GLES20.glUniformMatrix4fv(uTexMatrixLoc, 1, false, texMatrix, 0);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

            GLES20.glDisableVertexAttribArray(aPositionLoc);
            GLES20.glDisableVertexAttribArray(aTextureCoordLoc);

            EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurface, frameNs);
            EGL14.eglSwapBuffers(eglDisplay, eglSurface);
            armHeartbeat();
        } catch (Exception e) {
            Log.e(TAG, "Render failed", e);
        }
    }

    void setPaused(boolean paused) {
        this.paused = paused;
        if (paused) handler.removeCallbacks(heartbeat);
        else armHeartbeat();
    }

    // Turn to match how the phone was physically rotated, so both directions come out upright
    private float turnDegrees() {
        return displayRotation == Surface.ROTATION_270 ? 90f : -90f;
    }

    // Only Auto turns the picture; a locked orientation stays upright and is simply fitted
    private void computeMvp() {
        float upright = Math.min((float) outputWidth / sourceWidth, (float) outputHeight / sourceHeight);
        float turned = Math.min((float) outputWidth / sourceHeight, (float) outputHeight / sourceWidth);

        float uprightArea = (sourceWidth * upright) * (sourceHeight * upright);
        float turnedArea = (sourceHeight * turned) * (sourceWidth * turned);

        Matrix.setIdentityM(mvp, 0);
        if (allowRotation && turnedArea > uprightArea) {
            Matrix.rotateM(mvp, 0, turnDegrees(), 0f, 0f, 1f);
            Matrix.scaleM(mvp, 0,
                    sourceWidth * turned / outputHeight,
                    sourceHeight * turned / outputWidth, 1f);
        } else {
            Matrix.scaleM(mvp, 0,
                    sourceWidth * upright / outputWidth,
                    sourceHeight * upright / outputHeight, 1f);
        }
    }
}
