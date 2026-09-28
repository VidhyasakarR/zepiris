package com.loadshare.lsncapture

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.camera.lifecycle.ProcessCameraProvider
import com.google.android.gms.common.moduleinstall.ModuleInstall
import com.google.android.gms.common.moduleinstall.ModuleInstallRequest
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions

/**
 * The on-device parts, created once per process and shared by every capture.
 *
 * [warmUp] (the host calls it when the screen before the capture shows) makes
 * sure Play Services has the face model, loads it with one dummy frame, and
 * initialises CameraX, so the capture screen detects from its first frame.
 */
object FaceEngine {
    private const val TAG = "LsnCapture"

    val detector: FaceDetector by lazy {
        FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL) // eye-open probabilities
                .setMinFaceSize(0.12f)
                .enableTracking()
                .build()
        )
    }

    @Volatile var modelReady = false
        private set
    @Volatile var warmUpMs = -1L
        private set

    fun warmUp(context: Context, done: (Map<String, Any?>) -> Unit) {
        val t0 = SystemClock.elapsedRealtime()
        val app = context.applicationContext
        ProcessCameraProvider.getInstance(app) // CameraX init in the background
        val finish = { ready: Boolean, why: String? ->
            modelReady = ready
            warmUpMs = SystemClock.elapsedRealtime() - t0
            done(mapOf("modelReady" to ready, "ms" to warmUpMs, "reason" to why))
        }
        val runDummy = {
            detector.process(InputImage.fromBitmap(Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888), 0))
                .addOnSuccessListener { finish(true, null) }
                .addOnFailureListener { e -> Log.w(TAG, "face model not ready", e); finish(false, e.message) }
        }
        try {
            // Unbundled model: ask Play Services to install it now if it is missing.
            val client = ModuleInstall.getClient(app)
            client.areModulesAvailable(detector)
                .addOnSuccessListener { r ->
                    if (r.areModulesAvailable()) runDummy()
                    else client.installModules(ModuleInstallRequest.newBuilder().addApi(detector).build())
                        .addOnSuccessListener { runDummy() }
                        .addOnFailureListener { e -> finish(false, "model install failed: ${e.message}") }
                }
                .addOnFailureListener { runDummy() }
        } catch (e: Exception) {
            runDummy()
        }
    }
}
