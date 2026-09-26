package com.loadshare.lsncapture

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
class CaptureActivity : LsnBaseActivity() {
    companion object {
        const val EXTRA_CHALLENGE = "challenge"
        const val EXTRA_LIGHT = "light"           // start with the screen light (💡) on
        const val EXTRA_MAX_SIDE = "maxSide"      // long side of the photo in px; 0 = the camera's full resolution
        const val EXTRA_JPEG_QUALITY = "jpegQuality"
        const val EXTRA_BRIGHTNESS = "brightness" // starting exposure compensation, in EV (-2..+2)
        private const val MAX_BYTES = 4_500_000   // the server takes <= 5 MB per image
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
    private lateinit var freeze: android.widget.ImageView // the grabbed frame, shown the instant Capture is tapped

    // Which way round PreviewView.getBitmap() comes out on this phone (it can be
    // mirrored for the front camera). Measured once, at start, against an
    // analysis frame whose orientation is known (rotationDegrees, never mirrored).
    @Volatile private var previewMirrored: Boolean? = null
    @Volatile private var calibrating = false
    private var calibrationTries = 0
    private var previewUseCase: Preview? = null
    private var camera: androidx.camera.core.Camera? = null
    private lateinit var stage: FrameLayout
    private lateinit var rootView: FrameLayout
    private lateinit var bulb: Button
    private var lightOn = false

    /**
     * Screen light: the camera picture shrinks and the white around it, at full
     * screen brightness, lights the rider's face (a ring light). The camera's
     * auto-exposure adapts on its own; the rider then captures as usual.
     */
    private fun setLight(on: Boolean) {
        lightOn = on
        val bg = if (on) Color.WHITE else Color.rgb(5, 8, 15)
        rootView.setBackgroundColor(bg)
        // the bars around the fitted picture glow too (they were black: less light)
        preview.setBackgroundColor(bg)
        freeze.setBackgroundColor(bg)
        val k = if (on) 0.74f else 1f
        stage.animate().scaleX(k).scaleY(k).setDuration(180).start()
        bulb.background = pill(if (on) Color.rgb(255, 214, 90) else Color.argb(160, 15, 23, 40))
        // on: say plainly how to turn it off
        bulb.text = if (on) "💡 Turn off ✕" else "💡"
        bulb.textSize = if (on) 15f else 22f
        bulb.setTextColor(Color.rgb(20, 20, 20))
        bulb.contentDescription = if (on) "Turn the screen light off" else "Turn the screen light on"
        window.attributes = window.attributes.apply { screenBrightness = 1f }
        lightChip.text = if (on) "Low light — photo may be dark" else "Low light — tap 💡"
    }
    private var evIndex = 0
    private var evMin = 0
    private lateinit var brightRow: LinearLayout
    private lateinit var brightLabel: TextView
    private lateinit var brightSlider: android.widget.SeekBar

    /**
     * Camera brightness: exposure compensation, which brightens (or darkens) the
     * preview and therefore the photo itself. Starts at the host's setting
     * (config screen) and the rider can nudge it with − / + on screen.
     */
    private fun setupBrightness() {
        val exp = camera?.cameraInfo?.exposureState ?: return
        if (!exp.isExposureCompensationSupported) { brightRow.visibility = View.GONE; return }
        val step = exp.exposureCompensationStep.toFloat().takeIf { it > 0f } ?: 0.5f
        val ev = intent.getFloatExtra(EXTRA_BRIGHTNESS, 0f)
        evMin = exp.exposureCompensationRange.lower
        brightSlider.max = exp.exposureCompensationRange.upper - evMin
        setEv(Math.round(ev / step), exp)
        brightRow.visibility = View.VISIBLE
    }

    private fun setEv(index: Int, exp: androidx.camera.core.ExposureState? = camera?.cameraInfo?.exposureState) {
        exp ?: return
        val r = exp.exposureCompensationRange
        evIndex = index.coerceIn(r.lower, r.upper)
        camera?.cameraControl?.setExposureCompensationIndex(evIndex)
        val ev = evIndex * exp.exposureCompensationStep.toFloat()
        brightLabel.text = "☀ " + (if (ev > 0) "+" else "") + "%.1f".format(ev)
        if (brightSlider.progress != evIndex - evMin) brightSlider.progress = evIndex - evMin
    }

    /**
     * Upright width / height of the preview picture, from the camera's actual
     * preview resolution (3:4 asked for; a camera without it falls back to
     * another shape, and the crop and guide must follow).
     */
    private fun previewAspect(): Float {
        val r = previewUseCase?.resolutionInfo ?: return 3f / 4f
        val w = r.resolution.width.toFloat(); val h = r.resolution.height.toFloat()
        return if (r.rotationDegrees % 180 != 0) h / w else w / h
    }

    /** Where the fitted (FIT_CENTER) picture sits in a view of vw × vh. */
    private fun contentRect(vw: Float, vh: Float): RectF {
        val a = previewAspect()
        val pw = minOf(vw, vh * a); val ph = pw / a
        val x = (vw - pw) / 2; val y = (vh - ph) / 2
        return RectF(x, y, x + pw, y + ph)
    }

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
    // Armed between the tap and the grab.
    @Volatile private var steadying = false
    private var shotFired = false
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
        // full brightness: the screen light (💡) has to light the face
        window.attributes = window.attributes.apply { screenBrightness = 1f }
        challenge = Challenge(ChallengeType.from(intent.getStringExtra(EXTRA_CHALLENGE)))
        setContentView(buildUi())
        if (intent.getBooleanExtra(EXTRA_LIGHT, false)) setLight(true)
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
        // the camera picture, its guide and the frozen photo move together: the
        // screen light shrinks this stage so the white around it lights the face
        stage = FrameLayout(this)
        stage.addView(preview, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        guide = GuideView(this)
        stage.addView(guide, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        rootView = root
        root.addView(stage, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))

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
            text = "Checkpoint selfie"; setTextColor(Color.WHITE); textSize = 16f; paint.isFakeBoldText = true
            background = pill(Color.argb(160, 15, 23, 40)) // stays readable on the white light
            setPadding(dp(12f).toInt(), dp(8f).toInt(), dp(12f).toInt(), dp(8f).toInt())
        }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { leftMargin = dp(8f).toInt() })
        top.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f))
        lightChip = TextView(this).apply {
            text = "Low light — tap 💡"; setTextColor(Color.rgb(20, 20, 20)); textSize = 13f
            background = pill(Color.rgb(255, 214, 90)); setPadding(dp(12f).toInt(), dp(6f).toInt(), dp(12f).toInt(), dp(6f).toInt())
            visibility = View.GONE
        }
        top.addView(lightChip)
        bulb = Button(this).apply {
            text = "💡"; textSize = 22f; isAllCaps = false
            minWidth = dp(52f).toInt(); minimumWidth = dp(52f).toInt()
            setPadding(dp(12f).toInt(), 0, dp(12f).toInt(), 0)
            background = pill(Color.argb(160, 15, 23, 40))
            contentDescription = "Screen light"
            setOnClickListener { setLight(!lightOn) }
        }
        top.addView(bulb, LinearLayout.LayoutParams(WRAP_CONTENT, dp(52f).toInt()).apply { leftMargin = dp(8f).toInt() })
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
        // camera brightness (exposure compensation): brightens the preview and the photo
        brightRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; visibility = View.GONE
            background = pill(Color.argb(170, 15, 23, 40))
            setPadding(dp(14f).toInt(), dp(4f).toInt(), dp(10f).toInt(), dp(4f).toInt())
        }
        brightLabel = TextView(this).apply {
            text = "☀ 0"; setTextColor(Color.WHITE); textSize = 14f; gravity = Gravity.CENTER
            minWidth = dp(52f).toInt()
        }
        brightSlider = android.widget.SeekBar(this).apply {
            setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: android.widget.SeekBar, progress: Int, fromUser: Boolean) {
                    if (fromUser) setEv(evMin + progress)
                }
                override fun onStartTrackingTouch(sb: android.widget.SeekBar) {}
                override fun onStopTrackingTouch(sb: android.widget.SeekBar) {}
            })
        }
        brightRow.addView(brightLabel, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        brightRow.addView(brightSlider, LinearLayout.LayoutParams(0, dp(40f).toInt(), 1f))
        dock.addView(brightRow, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { bottomMargin = dp(8f).toInt() })
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

        freeze = android.widget.ImageView(this).apply {
            scaleType = android.widget.ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.rgb(5, 8, 15))
            visibility = View.GONE
        }
        stage.addView(freeze, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)) // over the preview, scaled with it
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
                val fps = fastAeRange(provider)
                val previewUc = Preview.Builder().setResolutionSelector(selector(1440, 1080))
                    .also { b -> fps?.let { aeRange(b, it) } }
                    .build()
                    .also { it.surfaceProvider = preview.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(selector(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, ::analyze); analysisUc = it }
                // 1920×1440 (portrait 1440×1920): plenty for face match and logo OCR, ~0.5-1 MB JPEG
                // Photo size from the host (config screen): long side maxSide, 4:3;
                // 0 = the camera's highest resolution. Quality mode: full ISP
                // processing (noise reduction, sharpening), not the fast path.
                val capSelector = if (maxSide <= 0) ResolutionSelector.Builder()
                    .setAspectRatioStrategy(fourByThree)
                    .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                    .build()
                else selector(maxSide, maxSide * 3 / 4)
                val cap = ImageCapture.Builder()
                    .setResolutionSelector(capSelector)
                    // Zero shutter lag where the camera supports it: the photo is the
                    // frame from the instant Capture was tapped (from a ring buffer),
                    // before the tap shakes the phone or the rider moves. Otherwise the
                    // fastest normal capture. (Quality mode merges several frames:
                    // slower, and any movement in between blurs.)
                    // fallback only (the photo is the preview frame, see grab()): no
                    // zero-shutter-lag ring buffer running all the time for nothing
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setJpegQuality(100) // the camera's JPEG is re-encoded once, at jpegQuality
                    .also { b -> fps?.let { aeRange(b, it) } }
                    .build()
                imageCapture = cap
                provider.unbindAll()
                previewUseCase = previewUc
                camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, previewUc, analysis, cap)
                setupBrightness()
                openedToBoundMs = SystemClock.elapsedRealtime() - openedAt
                setBanner("Fit your face in the oval", false)
            } catch (e: Exception) {
                Log.e(TAG, "camera start failed", e)
                fail("camera_unavailable: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * The camera's auto-exposure frame-rate range with the highest floor (>= 20 fps
     * if the camera offers one). A floor of 24 fps caps the shutter at ~1/24 s: in
     * dim light the camera raises the gain (a little grain) instead of leaving the
     * shutter open long enough to smear a moving hand or face.
     */
    @SuppressLint("UnsafeOptInUsageError")
    private fun fastAeRange(provider: ProcessCameraProvider): android.util.Range<Int>? = try {
        val info = provider.availableCameraInfos.firstOrNull { CameraSelector.DEFAULT_FRONT_CAMERA.filter(listOf(it)).isNotEmpty() }
        val ranges = info?.let {
            androidx.camera.camera2.interop.Camera2CameraInfo.from(it)
                .getCameraCharacteristic(android.hardware.camera2.CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
        }
        ranges?.filter { it.lower >= 20 }?.maxWithOrNull(compareBy({ it.lower }, { it.upper }))
    } catch (e: Exception) {
        null
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun zslSupported(provider: ProcessCameraProvider): Boolean = try {
        provider.availableCameraInfos.firstOrNull { CameraSelector.DEFAULT_FRONT_CAMERA.filter(listOf(it)).isNotEmpty() }
            ?.isZslSupported == true
    } catch (e: Throwable) {
        false
    }.also { zsl = it }

    private var zsl = false

    @SuppressLint("UnsafeOptInUsageError")
    private fun aeRange(b: androidx.camera.core.ExtendableBuilder<*>, r: android.util.Range<Int>) {
        androidx.camera.camera2.interop.Camera2Interop.Extender(b)
            .setCaptureRequestOption(android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, r)
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
        val frameLuma = meanLuma(proxy)
        val grid = lumaGrid(proxy) // small copy, read after detection (the frame is closed by then)
        // one comparison every ~15 frames until it is settled
        if (previewMirrored == null && !calibrating && frames > 8 && frames % 15 == 0) calibrate(proxy, rot)
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
                // Light is judged on the FACE (as the server's quality check does): a
                // dark room behind a well-lit face is fine. No face: the whole frame.
                val luma = samples.firstOrNull()?.let { faceLuma(grid, rot, it) } ?: frameLuma
                val shirt = samples.firstOrNull()?.let { shirtLuma(grid, rot, it) }
                onFaces(samples, luma, now, w, h, shirt)
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

    private val GW = 64
    private val GH = 48

    /** The luma plane down-sampled to GW × GH, in raw (sensor) orientation. */
    private fun lumaGrid(proxy: ImageProxy): IntArray {
        val plane = proxy.planes[0]
        val buf = plane.buffer
        val stride = plane.rowStride
        val g = IntArray(GW * GH)
        for (gy in 0 until GH) {
            val y = gy * proxy.height / GH
            for (gx in 0 until GW) {
                val x = gx * proxy.width / GW
                g[gy * GW + gx] = buf.get(y * stride + x).toInt() and 0xff
            }
        }
        return g
    }

    /** Mean luma inside the (upright, normalised) face box, read from the raw-orientation grid. */
    /** Grid luma at an upright, normalised point (null outside the frame). */
    private fun gridAt(grid: IntArray, rot: Int, ux: Float, uy: Float): Int? {
        val (rx, ry) = when (rot) {
            90 -> uy to 1f - ux
            180 -> 1f - ux to 1f - uy
            270 -> 1f - uy to ux
            else -> ux to uy
        }
        val gx = (rx * GW).toInt(); val gy = (ry * GH).toInt()
        return if (gx in 0 until GW && gy in 0 until GH) grid[gy * GW + gx] else null
    }

    /**
     * T-shirt light: the 75th percentile of the chin-to-stomach region (the
     * server's shirt region: +/-1.3 face widths, 1.3 to 3 face heights down).
     * A percentile, so a dark or black shirt in good light, with its lighter
     * folds and print, is not called "too dark" — that is the colour check's call.
     */
    private fun shirtLuma(grid: IntArray, rot: Int, f: FaceSample): Float? {
        val v = ArrayList<Int>(100)
        for (i in 0 until 10) for (j in 0 until 10) {
            val ux = f.cx + f.width * 1.3f * (-1f + 2f * i / 9f)
            val uy = f.top + f.height * (1.3f + 1.7f * j / 9f)
            if (uy < 1f && ux in 0f..1f) gridAt(grid, rot, ux, uy)?.let { v += it }
        }
        if (v.size < 12) return null // shirt barely in frame: the framing check handles that
        v.sort()
        return v[(v.size * 3) / 4].toFloat()
    }

    private fun faceLuma(grid: IntArray, rot: Int, f: FaceSample): Float {
        var sum = 0; var n = 0
        for (i in 0 until 10) for (j in 0 until 10) {
            // sample the inner part of the box (skip hair / background at the edges)
            val ux = f.left + f.width * (0.2f + 0.6f * i / 9f)
            val uy = f.top + f.height * (0.2f + 0.6f * j / 9f)
            // upright -> raw: undo the rotation that makes the frame upright
            val (rx, ry) = when (rot) {
                90 -> uy to 1f - ux
                180 -> 1f - ux to 1f - uy
                270 -> 1f - uy to ux
                else -> ux to uy
            }
            val gx = (rx * GW).toInt(); val gy = (ry * GH).toInt()
            if (gx in 0 until GW && gy in 0 until GH) { sum += grid[gy * GW + gx]; n++ }
        }
        return if (n == 0) 128f else sum.toFloat() / n
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

    private fun onFaces(faces: List<FaceSample>, luma: Float, now: Long, w: Int, h: Int, shirtLuma: Float? = null) {
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
        if (steadying) return // shot in progress: no UI changes
        val pending = !challenge.done
        val turning = pending && challenge.type == ChallengeType.TURN
        val blinking = pending && challenge.type == ChallengeType.BLINK
        val problem = liveProblem(faces, luma, shirtLuma, movement, turning, blinking)
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
        val low = luma < Limits.LOW_LIGHT || (shirtLuma != null && shirtLuma < Limits.LOW_LIGHT * 0.6f)
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
            lightChip.text = if (lightOn) "Low light — photo may be dark" else "Low light — tap 💡"
        }
    }

    // ---- UI state ---------------------------------------------------------------------
    private fun setBannerText(text: String) {
        if (banner.text != text) banner.text = text
    }

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
    // Flash is the host's choice (config screen): off by default. With it off, the
    // shot still waits for a still moment, which is what keeps it sharp.
    private var tapAt = 0L

    private fun capture() {
        if (!ready || capturing) return
        capturing = true
        captureBtn.isEnabled = false
        tapAt = SystemClock.elapsedRealtime()
        shotFired = false
        steadying = true // armed; grab disarms
        grab("tap") // zero delay: the frame on screen at the tap
    }

    /**
     * The photo is the preview frame on screen right now: no second capture, no
     * shutter, no wait. It is frozen on screen at once; encoding runs behind it.
     * (A full-resolution ImageCapture shot is only the fallback when the
     * preview cannot be read.)
     */
    private fun grab(why: String) {
        if (!steadying || shotFired) return
        shotFired = true
        steadying = false
        val shown = preview.bitmap // the frame PreviewView is showing, view-sized
        if (shown == null) {
            fireShot("$why/fallback"); return
        }
        tapToShotMs = SystemClock.elapsedRealtime() - tapAt
        val photo = previewContent(shown)
        freeze.setImageBitmap(photo)
        freeze.visibility = View.VISIBLE
        guide.visibility = View.GONE
        brightRow.visibility = View.GONE
        captureBtn.text = "✓"
        setBannerText("Got it")
        val t0 = SystemClock.elapsedRealtime()
        analysisExecutor.execute {
            val out = if (maxSide in 1 until max(photo.width, photo.height)) {
                val k = maxSide.toFloat() / max(photo.width, photo.height)
                android.graphics.Bitmap.createScaledBitmap(photo, (photo.width * k).toInt(), (photo.height * k).toInt(), true)
            } else photo
            val file = File(cacheDir, "lsn_capture_${System.currentTimeMillis()}.jpg")
            val te = SystemClock.elapsedRealtime()
            val (q, bytes) = writeJpeg(out, file)
            val encodeMs = SystemClock.elapsedRealtime() - te
            shotInfo = ",\"evIndex\":$evIndex,\"encodeMs\":$encodeMs,\"cameraOpenMs\":$openedToBoundMs,\"source\":\"preview\",\"previewMirrored\":$previewMirrored,\"width\":${out.width},\"height\":${out.height},\"jpegQuality\":$q,\"bytes\":$bytes"
            if (debug) Log.i(TAG, "grab: $why tap->frame ${tapToShotMs}ms $shotInfo")
            runOnUiThread { deliver(file, t0, lightOn, true) }
        }
    }

    /** The 3:4 picture inside the view-sized bitmap (FIT_CENTER bars cut off), the true way round. */
    private fun previewContent(b: android.graphics.Bitmap): android.graphics.Bitmap {
        val r = contentRect(b.width.toFloat(), b.height.toFloat())
        val x = r.left.toInt().coerceAtLeast(0); val y = r.top.toInt().coerceAtLeast(0)
        val m = android.graphics.Matrix()
        if (previewMirrored != false) m.postScale(-1f, 1f) // unknown yet: PreviewView mirrors the front camera by default
        return android.graphics.Bitmap.createBitmap(b, x, y, r.width().toInt().coerceAtMost(b.width - x), r.height().toInt().coerceAtMost(b.height - y), m, true)
    }

    /** Once: is the preview bitmap mirrored relative to the (known-good) analysis frame? */
    private fun calibrate(proxy: ImageProxy, rot: Int) {
        calibrating = true
        val truth = try {
            val raw = proxy.toBitmap()
            val m = android.graphics.Matrix().apply { postRotate(rot.toFloat()) }
            val up = android.graphics.Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
            tiny(up).also { if (up !== raw) up.recycle(); raw.recycle() }
        } catch (e: Exception) { calibrating = false; return }
        runOnUiThread {
            val shown = preview.bitmap
            if (shown == null) { calibrating = false; return@runOnUiThread }
            val r = contentRect(shown.width.toFloat(), shown.height.toFloat())
            val cx = r.left.toInt().coerceAtLeast(0); val cy = r.top.toInt().coerceAtLeast(0)
            val crop = android.graphics.Bitmap.createBitmap(shown, cx, cy,
                r.width().toInt().coerceAtMost(shown.width - cx), r.height().toInt().coerceAtMost(shown.height - cy))
            val p = tiny(crop)
            var straight = 0L; var flipped = 0L
            for (y in 0 until TH) for (x in 0 until TW) {
                val t = truth[y * TW + x]
                straight += abs(t - p[y * TW + x])
                flipped += abs(t - p[y * TW + (TW - 1 - x)])
            }
            // Decide only on a clear difference: a plain wall or a dark room looks
            // the same both ways round. Otherwise retry on a later frame.
            val lo = minOf(straight, flipped).toFloat(); val hi = maxOf(straight, flipped).toFloat()
            calibrationTries++
            if (hi > 0 && lo / hi < 0.8f) previewMirrored = flipped < straight
            if (debug) Log.i(TAG, "calibrate #$calibrationTries: straight=$straight flipped=$flipped -> previewMirrored=$previewMirrored")
            if (previewMirrored == null && calibrationTries < 20) calibrating = false // try again later
        }
    }

    private val TW = 24
    private val TH = 32
    private fun tiny(b: android.graphics.Bitmap): IntArray {
        val s = android.graphics.Bitmap.createScaledBitmap(b, TW, TH, true)
        val px = IntArray(TW * TH); s.getPixels(px, 0, TW, 0, 0, TW, TH); s.recycle()
        return IntArray(px.size) { (((px[it] shr 16) and 0xff) * 3 + ((px[it] shr 8) and 0xff) * 6 + (px[it] and 0xff)) / 10 }
    }

    private val maxSide by lazy { intent.getIntExtra(EXTRA_MAX_SIDE, 2592) }
    private val jpegQuality by lazy { intent.getIntExtra(EXTRA_JPEG_QUALITY, 92).coerceIn(60, 100) }
    private var shotInfo = ""
    private var tapToShotMs = -1L
    private var openedToBoundMs = -1L

    private fun fireShot(why: String) { // fallback: a full ImageCapture shot
        if (debug) Log.i(TAG, "shot: $why after ${SystemClock.elapsedRealtime() - tapAt} ms")
        val cap = imageCapture ?: return
        cap.flashMode = ImageCapture.FLASH_MODE_OFF
        val t0 = SystemClock.elapsedRealtime()
        setBannerText("Capturing…")
        captureBtn.text = "Capturing…"
        val shots = 1
        val taken = mutableListOf<android.graphics.Bitmap>()
        var rotationUsed = 0; var exifSeen = 0
        fun finishWith(list: List<android.graphics.Bitmap>) {
            setBannerText("Processing photo…")
            captureBtn.text = "Processing…"
            analysisExecutor.execute {
                val best = if (list.size == 1) list[0] else list.maxBy { sharpness(it) }
                list.filter { it !== best }.forEach { it.recycle() }
                val file = File(cacheDir, "lsn_capture_${System.currentTimeMillis()}.jpg")
                val (q, bytes) = writeJpeg(best, file)
                shotInfo = ",\"rotation\":$rotationUsed,\"exifOrientation\":$exifSeen,\"width\":${best.width},\"height\":${best.height},\"jpegQuality\":$q,\"bytes\":$bytes"
                best.recycle()
                runOnUiThread { deliver(file, t0, lightOn, true) }
            }
        }
        fun next() {
            cap.takePicture(analysisExecutor, object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    // Straighten with the camera's own rotation for this frame (the same
                    // value the live face detection uses, which finds faces upright), not
                    // the EXIF tag: on some front cameras that tag is 180° off, which
                    // turned photos upside down. Never mirrored: the true scene.
                    if (tapToShotMs < 0) tapToShotMs = SystemClock.elapsedRealtime() - tapAt
                    val rot = image.imageInfo.rotationDegrees
                    val buf = image.planes[0].buffer
                    val jpeg = ByteArray(buf.remaining()).also { buf.get(it) }
                    image.close()
                    rotationUsed = rot
                    exifSeen = try {
                        androidx.exifinterface.media.ExifInterface(java.io.ByteArrayInputStream(jpeg))
                            .getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION, 0)
                    } catch (_: Exception) { -1 }
                    val bmp = uprightBitmap(jpeg, rot)
                    if (bmp != null) taken += bmp
                    if (taken.size < shots && bmp != null) runOnUiThread { next() }
                    else runOnUiThread { if (taken.isEmpty()) failShot() else finishWith(taken.toList()) }
                }

                override fun onError(e: ImageCaptureException) {
                    Log.e(TAG, "capture failed", e)
                    runOnUiThread { if (taken.isEmpty()) failShot() else finishWith(taken.toList()) }
                }
            })
        }
        next()
    }

    private fun failShot() {
        capturing = false
        setBanner("Capture failed — try again", false)
    }

    /** Decode the camera's JPEG (pixels only, EXIF ignored), rotate upright, cap the long side. */
    private fun uprightBitmap(jpeg: ByteArray, rot: Int): android.graphics.Bitmap? {
        val bounds = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
        android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, bounds)
        val long = max(bounds.outWidth, bounds.outHeight)
        var sample = 1
        if (maxSide > 0) while (long / (sample * 2) >= maxSide) sample *= 2
        val src = android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size,
            android.graphics.BitmapFactory.Options().apply { inSampleSize = sample }) ?: return null
        val m = android.graphics.Matrix()
        if (rot != 0) m.postRotate(rot.toFloat())
        val srcLong = max(src.width, src.height)
        if (maxSide in 1 until srcLong) {
            val k = maxSide.toFloat() / srcLong
            m.postScale(k, k)
        }
        if (m.isIdentity) return src
        val out = android.graphics.Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        if (out !== src) src.recycle()
        return out
    }

    /** Encode at the requested quality, stepping down only if the file would pass the server's limit. */
    private fun writeJpeg(bmp: android.graphics.Bitmap, file: File): Pair<Int, Int> {
        var q = jpegQuality
        while (true) {
            val bos = java.io.ByteArrayOutputStream()
            bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, q, bos)
            if (bos.size() <= MAX_BYTES || q <= 70) {
                file.writeBytes(bos.toByteArray())
                return q to bos.size()
            }
            q -= 5
        }
    }

    /** Sharpness of the upper-centre (face) region: variance of a Laplacian on a small grey copy. */
    private fun sharpness(full: android.graphics.Bitmap): Double {
        val bmp = android.graphics.Bitmap.createScaledBitmap(full, max(1, full.width / 4), max(1, full.height / 4), true)
        val x0 = bmp.width / 4; val x1 = bmp.width * 3 / 4
        val y0 = bmp.height / 10; val y1 = bmp.height * 11 / 20
        val w = x1 - x0; val h = y1 - y0
        if (w < 3 || h < 3) return 0.0
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, x0, y0, w, h)
        bmp.recycle()
        val g = IntArray(px.size) { (((px[it] shr 16) and 0xff) * 3 + ((px[it] shr 8) and 0xff) * 6 + (px[it] and 0xff)) / 10 }
        var sum = 0.0; var sq = 0.0; var n = 0
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            val lap = (4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w]).toDouble()
            sum += lap; sq += lap * lap; n++
        }
        if (n == 0) return 0.0
        val mean = sum / n
        return sq / n - mean * mean
    }

    private fun deliver(file: File, t0: Long, light: Boolean, upright: Boolean) {
                        val stats = "{\"challenge\":\"${challenge.type.name.lowercase()}\",\"challengePassed\":${challenge.done}," +
                            "\"light\":$light,\"firstDetectionMs\":$firstDetectionMs,\"captureMs\":${SystemClock.elapsedRealtime() - t0}," +
                            "\"detectMsAvg\":${if (frames > 0) detectMsTotal / frames else -1},\"frames\":$frames,\"upright\":$upright,\"zsl\":$zsl,\"tapToShotMs\":$tapToShotMs$shotInfo}"
        setResult(RESULT_OK, Intent().putExtra(RESULT_PATH, file.absolutePath).putExtra(RESULT_JSON, stats))
        finish()
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
            val cr = contentRect(vw, vh)
            val pw = cr.width(); val ph = cr.height()
            val px = cr.left; val py = cr.top
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
