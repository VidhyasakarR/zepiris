package com.loadshare.lsn_capture_sdk

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.activity.addCallback
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.google.mlkit.vision.common.InputImage
import java.io.File
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import kotlin.math.abs
import kotlin.math.max

/**
 * The native capture screen: front camera, oval guide, live checks, liveness
 * challenge, capture. Returns the photo file path (and what happened) to
 * [LsnCaptureSdkPlugin]; review / quality check / submit happen in Dart.
 *
 * Speed and accuracy come from CameraX use cases at their own resolutions:
 * - Preview and photo: 4:3, the camera's native shape (the phone's own camera
 *   app framing). The preview is fitted, not cropped, so the rider sees exactly
 *   what the photo will contain.
 * - Analysis: 640×480, same 4:3 field of view, straight to ML Kit on a worker
 *   thread: no frame copies, ~20-30 detections/s, so a 150 ms blink is caught.
 */
class CaptureActivity : ComponentActivity() {
    companion object {
        const val EXTRA_CHALLENGE = "challenge"
        const val RESULT_PATH = "path"
        const val RESULT_JSON = "result"
        private const val TAG = "LsnCapture"
        private const val HOLD_MS = 400L
        private const val MODEL_WAIT_MS = 10_000L
    }

    private lateinit var preview: PreviewView
    private lateinit var guide: GuideView
    private lateinit var banner: TextView
    private lateinit var captureBtn: Button
    private lateinit var lightChip: TextView
    private lateinit var flash: View

    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    // ML Kit listeners can fire after onDestroy shut the executor down: run them
    // inline then, so a late result never throws and every frame still closes.
    private val listenerExecutor = Executor { r ->
        if (analysisExecutor.isShutdown) r.run()
        else try { analysisExecutor.execute(r) } catch (_: RejectedExecutionException) { r.run() }
    }
    private var analysisUc: ImageAnalysis? = null
    private var imageCapture: ImageCapture? = null
    private lateinit var challenge: Challenge

    // written on the analysis thread, read on the main thread through post()
    @Volatile private var ready = false
    @Volatile private var lastText = ""
    private var lowLight = false
    @Volatile private var capturing = false
    private var alignedAt = 0L
    private var faceSeenAt = 0L
    private val trail = ArrayDeque<Triple<Long, Float, Float>>()
    private var openedAt = 0L
    @Volatile private var firstDetectionMs = -1L
    private var firstFailureAt = 0L
    private var frames = 0
    private var detectMsTotal = 0L
    private val debug by lazy { (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0 || intent.getBooleanExtra("debug", false) }
    private var lastLog = 0L

    // ---- setup -------------------------------------------------------------------
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        openedAt = SystemClock.elapsedRealtime()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // full brightness: the low-light screen flash has to light the face
        window.attributes = window.attributes.apply { screenBrightness = 1f }
        challenge = Challenge(ChallengeType.from(intent.getStringExtra(EXTRA_CHALLENGE)))
        setContentView(buildUi())
        onBackPressedDispatcher.addCallback(this) { cancel() } // predictive back (API 33+)
        cleanOldPhotos()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            fail("no_permission")
            return
        }
        startCamera()
    }

    override fun onDestroy() {
        analysisUc?.clearAnalyzer() // no new frames into a closing screen
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    /** Temp selfies left by an earlier run (process death, back mid-capture). */
    private fun cleanOldPhotos() {
        val cutoff = System.currentTimeMillis() - 10 * 60_000L
        cacheDir.listFiles { f -> f.name.startsWith("lsn_capture_") && f.lastModified() < cutoff }
            ?.forEach { it.delete() }
    }

    private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

    private fun pill(color: Int) = GradientDrawable().apply { cornerRadius = dp(22f); setColor(color) }

    private fun buildUi(): View {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.rgb(5, 8, 15)) }
        preview = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER // show the whole 4:3 frame = the photo
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            // PreviewView mirrors the front camera like a mirror; flip it back so
            // the rider sees the true scene, the same as the photo that is sent
            // (T-shirt text reads normally). COMPATIBLE mode (a TextureView) is
            // what makes this view-level flip work.
            scaleX = -1f
        }
        root.addView(preview, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        guide = GuideView(this)
        root.addView(guide, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))

        val top = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(8f).toInt(), dp(36f).toInt(), dp(12f).toInt(), 0)
        }
        top.addView(Button(this).apply {
            text = "←"; textSize = 22f; setTextColor(Color.WHITE); background = pill(Color.argb(140, 15, 23, 40))
            setOnClickListener { cancel() }
        }, LinearLayout.LayoutParams(dp(52f).toInt(), dp(52f).toInt()))
        top.addView(TextView(this).apply {
            text = "  Checkpoint selfie"; setTextColor(Color.WHITE); textSize = 17f; paint.isFakeBoldText = true
        }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        lightChip = TextView(this).apply {
            text = "Low light — screen will flash"; setTextColor(Color.rgb(20, 20, 20)); textSize = 13f
            background = pill(Color.WHITE); setPadding(dp(12f).toInt(), dp(6f).toInt(), dp(12f).toInt(), dp(6f).toInt())
            visibility = View.GONE
        }
        top.addView(lightChip)
        root.addView(top, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.TOP))

        val dock = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16f).toInt(), 0, dp(16f).toInt(), dp(28f).toInt())
        }
        banner = TextView(this).apply {
            text = "Starting the camera…"; setTextColor(Color.WHITE); textSize = 16f; gravity = Gravity.CENTER
            paint.isFakeBoldText = true; background = pill(Color.argb(210, 15, 23, 40))
            setPadding(dp(14f).toInt(), dp(10f).toInt(), dp(14f).toInt(), dp(10f).toInt())
        }
        dock.addView(banner, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { bottomMargin = dp(10f).toInt() })
        captureBtn = Button(this).apply {
            text = "Capture — align first"; textSize = 17f; isAllCaps = false; setTextColor(Color.argb(200, 255, 255, 255))
            background = pill(Color.rgb(38, 50, 77)); isEnabled = false
            setOnClickListener { capture() }
        }
        dock.addView(captureBtn, LinearLayout.LayoutParams(MATCH_PARENT, dp(56f).toInt()))
        root.addView(dock, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))
        // Android 15+ draws edge-to-edge: keep the top bar and the Capture button
        // clear of the status bar, notch and navigation bar.
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            top.updatePadding(top = b.top + dp(8f).toInt())
            dock.updatePadding(bottom = b.bottom + dp(16f).toInt())
            insets
        }

        flash = View(this).apply { setBackgroundColor(Color.WHITE); visibility = View.GONE }
        root.addView(flash, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        return root
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val fourByThree = AspectRatioStrategy(androidx.camera.core.AspectRatio.RATIO_4_3, AspectRatioStrategy.FALLBACK_RULE_AUTO)
                fun selector(w: Int, h: Int) = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(fourByThree)
                    .setResolutionStrategy(ResolutionStrategy(Size(w, h), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                    .build()
                val previewUc = Preview.Builder().setResolutionSelector(selector(1440, 1080)).build()
                    .also { it.surfaceProvider = preview.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(selector(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, ::analyze); analysisUc = it }
                // 1920×1440 (portrait 1440×1920): plenty for face match and logo OCR, ~0.5-1 MB JPEG
                val cap = ImageCapture.Builder()
                    .setResolutionSelector(selector(1920, 1440))
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setJpegQuality(90)
                    .build()
                imageCapture = cap
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, previewUc, analysis, cap)
                setBanner("Fit your face in the oval", false)
            } catch (e: Exception) {
                Log.e(TAG, "camera start failed", e)
                fail("camera_unavailable: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ---- per frame (analysis thread) ----------------------------------------------
    @SuppressLint("UnsafeOptInUsageError")
    private fun analyze(proxy: ImageProxy) {
        val media = proxy.image
        if (media == null || capturing || isFinishing) {
            proxy.close(); return
        }
        val rot = proxy.imageInfo.rotationDegrees
        val w = if (rot % 180 == 0) proxy.width else proxy.height // upright frame size
        val h = if (rot % 180 == 0) proxy.height else proxy.width
        val luma = meanLuma(proxy)
        val t0 = SystemClock.elapsedRealtime()
        FaceEngine.detector.process(InputImage.fromMediaImage(media, rot))
            .addOnSuccessListener(listenerExecutor) { faces ->
                val now = SystemClock.elapsedRealtime()
                frames++; detectMsTotal += now - t0
                if (firstDetectionMs < 0) firstDetectionMs = now - openedAt
                val samples = faces.map { f ->
                    val b = f.boundingBox
                    FaceSample(
                        b.left / w.toFloat(), b.top / h.toFloat(), b.width() / w.toFloat(), b.height() / h.toFloat(),
                        f.headEulerAngleY, f.headEulerAngleX, f.headEulerAngleZ,
                        f.leftEyeOpenProbability, f.rightEyeOpenProbability,
                    )
                }
                onFaces(samples, luma, now, w, h)
            }
            .addOnFailureListener(listenerExecutor) { e ->
                // Play Services model not there yet ("waiting for the face module"): say so
                // and keep trying; give up clearly if it never comes.
                val now = SystemClock.elapsedRealtime()
                if (firstFailureAt == 0L) firstFailureAt = now
                if (debug || firstFailureAt == now) Log.w(TAG, "detection failed: ${e.message}")
                runOnUiThread {
                    if (firstDetectionMs < 0 && now - firstFailureAt > MODEL_WAIT_MS) fail("face_check_unavailable")
                    else if (!capturing) setBanner("Preparing face check…", false)
                }
            }
            .addOnCompleteListener(listenerExecutor) { proxy.close() }
    }

    private fun meanLuma(proxy: ImageProxy): Float {
        val plane = proxy.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        var sum = 0L; var n = 0
        var y = 0
        while (y < proxy.height) {
            var x = 0
            while (x < proxy.width) {
                sum += buf.get(y * rowStride + x).toInt() and 0xff; n++
                x += 16
            }
            y += 16
        }
        return if (n == 0) 128f else sum.toFloat() / n
    }

    private fun onFaces(faces: List<FaceSample>, luma: Float, now: Long, w: Int, h: Int) {
        if (faces.isNotEmpty()) faceSeenAt = now
        // a different person could step in: after 1.2 s with no face, or a second face
        if ((faces.isEmpty() && now - faceSeenAt > 1200 && (challenge.done || challenge.phase > 0)) || faces.size > 1) {
            challenge.reset()
        }
        var movement = 0f
        if (faces.size == 1) {
            val f = faces[0]
            trail.addLast(Triple(now, f.cx, f.cy))
            while (trail.isNotEmpty() && now - trail.first().first > 400) trail.removeFirst()
            for ((_, x, y) in trail) movement = max(movement, abs(x - f.cx) + abs(y - f.cy))
        }
        val pending = !challenge.done
        val turning = pending && challenge.type == ChallengeType.TURN
        val blinking = pending && challenge.type == ChallengeType.BLINK
        val problem = liveProblem(faces, luma, movement, turning, blinking)
        // count the challenge on every single-face frame that is roughly frontal
        // (a blink during a framing flicker still counts)
        if (pending && faces.size == 1 && (turning || (abs(faces[0].yaw) < 18 && abs(faces[0].pitch) < 18))) {
            challenge.step(faces[0])
        }
        val text: String
        val ok: Boolean
        when {
            problem != null -> {
                val held = now - alignedAt < HOLD_MS && ready
                text = if (held) lastText else problem
                ok = held
            }
            !challenge.done -> { text = challenge.instruction; ok = false }
            else -> { alignedAt = now; text = "Perfect — tap Capture"; ok = true }
        }
        val low = luma < Limits.LOW_LIGHT
        if (debug && now - lastLog > 1000) {
            lastLog = now
            val f = faces.firstOrNull()
            Log.i(TAG, "frame ${w}x$h fps~${if (now > openedAt) frames * 1000 / (now - openedAt) else 0} detect=${if (frames > 0) detectMsTotal / frames else 0}ms " +
                "first=${firstDetectionMs}ms luma=${luma.toInt()} " + (f?.let {
                    "face l=%.2f t=%.2f w=%.2f h=%.2f yaw=%.0f pitch=%.0f eyes=%.2f/%.2f shirt=%.2f".format(
                        it.left, it.top, it.width, it.height, it.yaw, it.pitch, it.leftEyeOpen ?: -1f, it.rightEyeOpen ?: -1f, shirtVisible(it))
                } ?: "no face") + " -> ${problem ?: "ok"} challenge=${challenge.type}/${challenge.phase}/${challenge.done}")
        }
        lastText = text
        runOnUiThread {
            if (capturing) return@runOnUiThread
            ready = ok
            lowLight = low
            setBanner(text, ok)
            lightChip.visibility = if (low) View.VISIBLE else View.GONE
        }
    }

    // ---- UI state ---------------------------------------------------------------------
    private fun setBanner(text: String, ok: Boolean) {
        if (banner.text != text) banner.text = text
        banner.background = pill(if (ok) Color.rgb(15, 138, 95) else Color.argb(210, 15, 23, 40))
        guide.ok = ok
        captureBtn.isEnabled = ok
        captureBtn.text = if (ok) "Capture" else "Capture — align first"
        captureBtn.setTextColor(if (ok) Color.WHITE else Color.argb(200, 255, 255, 255))
        captureBtn.background = pill(if (ok) Color.rgb(15, 138, 95) else Color.rgb(38, 50, 77))
    }

    // ---- capture --------------------------------------------------------------------------
    private fun capture() {
        val cap = imageCapture ?: return
        if (!ready || capturing) return
        capturing = true
        captureBtn.isEnabled = false
        captureBtn.text = "Capturing…"
        val useFlash = lowLight
        val shoot = {
            val file = File(cacheDir, "lsn_capture_${System.currentTimeMillis()}.jpg")
            val t0 = SystemClock.elapsedRealtime()
            cap.takePicture(ImageCapture.OutputFileOptions.Builder(file).build(), ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                        flash.visibility = View.GONE
                        // CameraX saves the sensor's pixels plus an EXIF rotation tag
                        // (sideways on most front cameras). Bake the rotation into the
                        // pixels, off the main thread, so every viewer and the server see
                        // the same upright photo without relying on EXIF handling.
                        analysisExecutor.execute {
                            val ok = try { uprightJpeg(file) } catch (e: Exception) { Log.w(TAG, "upright failed", e); false }
                            runOnUiThread { deliver(file, t0, useFlash, ok) }
                        }
                    }

                    override fun onError(e: ImageCaptureException) {
                        Log.e(TAG, "capture failed", e)
                        flash.visibility = View.GONE
                        capturing = false
                        setBanner("Capture failed — try again", false)
                    }
                })
        }
        if (useFlash) {
            // light the face with the screen and give auto-exposure time to adjust
            flash.visibility = View.VISIBLE
            flash.postDelayed(shoot, 800)
        } else {
            shoot()
        }
    }

    private fun deliver(file: File, t0: Long, useFlash: Boolean, upright: Boolean) {
                        val stats = "{\"challenge\":\"${challenge.type.name.lowercase()}\",\"challengePassed\":${challenge.done}," +
                            "\"flashed\":$useFlash,\"firstDetectionMs\":$firstDetectionMs,\"captureMs\":${SystemClock.elapsedRealtime() - t0}," +
                            "\"detectMsAvg\":${if (frames > 0) detectMsTotal / frames else -1},\"frames\":$frames,\"upright\":$upright}"
        setResult(RESULT_OK, Intent().putExtra(RESULT_PATH, file.absolutePath).putExtra(RESULT_JSON, stats))
        finish()
    }

    /**
     * Rewrite [file] with its EXIF orientation applied to the pixels (and the
     * tag reset). The photo stays the true scene, not mirrored: text on the
     * T-shirt reads normally, which the logo check relies on. Returns true when
     * the file is upright afterwards.
     */
    private fun uprightJpeg(file: File): Boolean {
        val exif = androidx.exifinterface.media.ExifInterface(file.absolutePath)
        val o = exif.getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION, androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL)
        val m = android.graphics.Matrix()
        when (o) {
            androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL, androidx.exifinterface.media.ExifInterface.ORIENTATION_UNDEFINED -> return true
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            androidx.exifinterface.media.ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
            else -> return true
        }
        val src = android.graphics.BitmapFactory.decodeFile(file.absolutePath) ?: return false
        val out = android.graphics.Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        if (out !== src) src.recycle()
        java.io.FileOutputStream(file).use { out.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, it) }
        out.recycle()
        return true // a fresh JPEG carries no orientation tag
    }

    private fun cancel() {
        setResult(RESULT_CANCELED)
        finish()
    }

    private fun fail(why: String) {
        if (isFinishing) return
        setResult(RESULT_FIRST_USER, Intent().putExtra("error", why))
        finish()
    }

    /** Face oval and shoulders, drawn over the fitted 4:3 picture. */
    private inner class GuideView(ctx: android.content.Context) : View(ctx) {
        var ok = false
            set(v) { if (field != v) { field = v; invalidate() } }
        private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }

        override fun onDraw(c: Canvas) {
            // the picture is fitted: a 3:4 rect centred in the view
            val vw = width.toFloat(); val vh = height.toFloat()
            val pw = minOf(vw, vh * 3f / 4f); val ph = pw * 4f / 3f
            val px = (vw - pw) / 2; val py = (vh - ph) / 2
            paint.strokeWidth = dp(if (ok) 4f else 2.5f)
            paint.color = if (ok) Color.rgb(60, 207, 149) else Color.argb(220, 255, 255, 255)
            val face = RectF(px + pw * 0.33f, py + ph * 0.10f, px + pw * 0.67f, py + ph * 0.42f)
            c.drawOval(face, paint)
            val top = face.bottom + ph * 0.03f
            val body = Path().apply {
                moveTo(px + pw * 0.08f, py + ph)
                lineTo(px + pw * 0.08f, top + ph * 0.16f)
                quadTo(px + pw * 0.10f, top + ph * 0.01f, px + pw * 0.36f, top)
                lineTo(px + pw * 0.64f, top)
                quadTo(px + pw * 0.90f, top + ph * 0.01f, px + pw * 0.92f, top + ph * 0.16f)
                lineTo(px + pw * 0.92f, py + ph)
            }
            c.drawPath(body, paint)
        }
    }
}
