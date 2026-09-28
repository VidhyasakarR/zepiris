package com.loadshare.lsncapture

import android.app.Activity
import android.app.Application
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import com.facebook.react.bridge.ActivityEventListener
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.ReadableType
import com.facebook.react.bridge.UiThreadUtil
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import org.json.JSONArray
import org.json.JSONObject
import java.util.WeakHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * React Native bridge (legacy module, works through the new-arch interop layer):
 * - `warmUp`: load the face model + CameraX ahead of time ([FaceEngine]).
 * - `capture`: run [CaptureActivity], resolve with the photo path and stats.
 * - `score`: POST the photo to `/v1/checkpoint/score` ([ScoreClient]).
 *
 * Promise reject codes are the JS `LsnErrorCode` strings.
 * The JPEG is NOT deleted after capture: the host uploads it (e.g. to S3);
 * [CaptureActivity] removes photos older than 10 minutes on its next run.
 *
 * Capture lifecycle:
 * - One capture at a time (`busy`). The pending promise is settled exactly once:
 *   by the activity result, by a failed launch, or (see [captureWatcher]) when
 *   the capture screen closes without its result reaching us, so the module can
 *   never stay stuck on `busy`.
 * - The host activity being destroyed and recreated while the capture screen is
 *   open (low memory, "Don't keep activities") is fine: the React context lives
 *   on and the result is delivered to the recreated activity.
 * - Process death while the capture screen is open: the JS promise dies with the
 *   old runtime. The new process gets the activity result before React is up, so
 *   nobody receives it; the orphan photo is deleted by the next capture's cleanup.
 * - React context torn down (reload, host destroyed): [invalidate] drops the
 *   pending promise (its JS runtime is gone, there is nobody to reject to).
 */
class LsnCaptureModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext), ActivityEventListener {

    companion object {
        const val NAME = "LsnCapture"
        /** Same request code as the Flutter plugin; fits the 16 bits FragmentActivity allows. */
        private const val REQUEST = 0x15C4
        /** Grace after the capture screen is destroyed before an undelivered result counts as lost. */
        private const val ORPHAN_GRACE_MS = 1_500L
        private val CHALLENGES = setOf("none", "blink", "turn", "random")
    }

    private val lock = Any()
    private var pending: Promise? = null
    /** Increments per capture, so late callbacks never settle a newer capture. */
    private var launchId = 0
    private val main = Handler(Looper.getMainLooper())
    private val executor: ExecutorService = Executors.newCachedThreadPool()

    /**
     * Watches CaptureActivity instances opened for a pending capture. If one
     * finishes and, after [ORPHAN_GRACE_MS], its result still has not reached
     * [onActivityResult] (e.g. a host activity that does not forward results to
     * React), the capture rejects with `capture_failed` instead of leaving the
     * module `busy` until the app restarts. Normally the result arrives before
     * the finishing activity is destroyed, so this never fires.
     */
    private val openedFor = WeakHashMap<Activity, Int>() // main thread only
    private val captureWatcher = object : Application.ActivityLifecycleCallbacks {
        override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {
            if (activity !is CaptureActivity) return
            val id = synchronized(lock) { if (pending != null) launchId else null } ?: return
            openedFor[activity] = id
        }

        override fun onActivityDestroyed(activity: Activity) {
            if (activity !is CaptureActivity) return
            val id = openedFor.remove(activity) ?: return
            if (activity.isChangingConfigurations) return // recreated: the new instance is watched instead
            main.postDelayed({
                take(id)?.reject("capture_failed", "the capture screen closed without returning a result")
            }, ORPHAN_GRACE_MS)
        }

        override fun onActivityStarted(activity: Activity) {}
        override fun onActivityResumed(activity: Activity) {}
        override fun onActivityPaused(activity: Activity) {}
        override fun onActivityStopped(activity: Activity) {}
        override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) {}
    }

    init {
        reactContext.addActivityEventListener(this)
        (reactContext.applicationContext as? Application)?.registerActivityLifecycleCallbacks(captureWatcher)
    }

    override fun getName(): String = NAME

    override fun invalidate() {
        reactContext.removeActivityEventListener(this)
        (reactContext.applicationContext as? Application)?.unregisterActivityLifecycleCallbacks(captureWatcher)
        synchronized(lock) { pending = null; launchId++ }
        main.removeCallbacksAndMessages(null)
        executor.shutdown() // in-flight score calls finish (bounded by their timeoutMs)
        super.invalidate()
    }

    /** Claims the pending promise (only the one for [id], when given). */
    private fun take(id: Int? = null): Promise? = synchronized(lock) {
        if (id != null && id != launchId) null else pending.also { pending = null }
    }

    // ---- warmUp ------------------------------------------------------------------------
    /** Never rejects: failures resolve with modelReady=false and a reason. Can take long (model download). */
    @ReactMethod
    fun warmUp(promise: Promise) {
        try {
            FaceEngine.warmUp(reactContext) { r ->
                promise.resolve(Arguments.createMap().apply {
                    putBoolean("modelReady", r["modelReady"] as? Boolean ?: false)
                    putDouble("ms", (r["ms"] as? Number)?.toDouble() ?: -1.0)
                    val reason = r["reason"] as? String
                    if (reason != null) putString("reason", reason) else putNull("reason")
                })
            }
        } catch (e: Exception) {
            promise.resolve(Arguments.createMap().apply {
                putBoolean("modelReady", false)
                putDouble("ms", -1.0)
                putString("reason", e.message ?: e.javaClass.simpleName)
            })
        }
    }

    // ---- capture -----------------------------------------------------------------------
    @ReactMethod
    fun capture(opts: ReadableMap?, promise: Promise) {
        if (reactContext.currentActivity == null) return promise.reject("no_activity", "no foreground activity")
        // Unknown / miscased challenge → the default 'blink' (CaptureActivity would silently run none).
        val challenge = opts.str("challenge")?.trim()?.lowercase()?.takeIf { it in CHALLENGES } ?: "blink"
        val maxSide = opts.finite("maxSide")?.toInt() ?: 2592
        val jpegQuality = opts.finite("jpegQuality")?.toInt() ?: 92
        val brightness = (opts.finite("brightness") ?: 0.0).coerceIn(-2.0, 2.0).toFloat()
        val light = opts.bool("light") ?: false
        val id = synchronized(lock) {
            if (pending != null) return promise.reject("busy", "a capture is already running")
            pending = promise
            ++launchId
        }
        // Launch on the main thread (the bridge calls us on the native-modules thread).
        UiThreadUtil.runOnUiThread {
            val activity = reactContext.currentActivity
            if (activity == null || activity.isFinishing) {
                take(id)?.reject("no_activity", "no foreground activity")
                return@runOnUiThread
            }
            try {
                activity.startActivityForResult(
                    Intent(activity, CaptureActivity::class.java)
                        .putExtra(CaptureActivity.EXTRA_CHALLENGE, challenge)
                        .putExtra(CaptureActivity.EXTRA_LIGHT, light)
                        .putExtra(CaptureActivity.EXTRA_MAX_SIDE, maxSide)
                        .putExtra(CaptureActivity.EXTRA_JPEG_QUALITY, jpegQuality)
                        .putExtra(CaptureActivity.EXTRA_BRIGHTNESS, brightness),
                    REQUEST,
                )
            } catch (e: Exception) { // never leave the module stuck on "busy"
                take(id)?.reject("capture_failed", e.message ?: "could not open the capture screen", e)
            }
        }
    }

    override fun onActivityResult(activity: Activity, requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode != REQUEST) return
        val p = take() ?: return
        when (resultCode) {
            Activity.RESULT_OK -> {
                val path = data?.getStringExtra(CaptureActivity.RESULT_PATH)
                if (path.isNullOrEmpty()) {
                    p.reject("capture_failed", "the capture screen returned no photo")
                    return
                }
                val stats = try {
                    JSONObject(data?.getStringExtra(CaptureActivity.RESULT_JSON) ?: "{}")
                } catch (e: Exception) {
                    JSONObject()
                }
                p.resolve(Arguments.createMap().apply {
                    putString("path", path)
                    putString("uri", "file://$path")
                    putString("challenge", stats.optString("challenge", "none"))
                    putBoolean("challengePassed", stats.optBoolean("challengePassed", false))
                    putMap("stats", toWritableMap(stats))
                })
            }
            Activity.RESULT_CANCELED -> p.reject("cancelled", "the rider backed out")
            else -> {
                val why = data?.getStringExtra("error") ?: "capture_failed"
                val code = when {
                    why.startsWith("no_permission") -> "no_permission"
                    why.startsWith("face_check_unavailable") -> "face_check_unavailable"
                    else -> "capture_failed" // e.g. "camera_unavailable: <reason>"
                }
                p.reject(code, why)
            }
        }
    }

    override fun onNewIntent(intent: Intent) {} // CaptureActivity is started for a result, never via a new intent

    // ---- score -------------------------------------------------------------------------
    @ReactMethod
    fun score(opts: ReadableMap?, promise: Promise) {
        val req = try {
            ScoreRequest.validate(
                apiBase = opts.str("apiBase"),
                checks = opts.strList("checks"),
                faceCheckPath = opts.str("faceCheckPath"),
                faceCheckS3 = opts.str("faceCheckS3"),
                sourceSelfieS3 = opts.str("sourceSelfieS3"),
                sourceSelfieB64 = opts.str("sourceSelfieB64"),
                timeoutMs = opts.num("timeoutMs")?.let { if (it.isFinite()) it.toLong() else -1L },
                headers = opts.strMap("headers"),
            )
        } catch (e: ScoreException) {
            return promise.reject(e.code, e.message)
        } catch (e: Exception) {
            return promise.reject("invalid_config", e.message ?: "invalid score options")
        }
        try {
            executor.execute {
                try {
                    promise.resolve(toWritableMap(ScoreClient.score(req)))
                } catch (e: ScoreException) {
                    promise.reject(e.code, e.message)
                } catch (e: Throwable) { // incl. OutOfMemoryError: the promise must always settle
                    promise.reject("network", e.message ?: e.javaClass.simpleName, e)
                }
            }
        } catch (e: Exception) { // executor shut down (module invalidated)
            promise.reject("network", e.message ?: "score executor unavailable", e)
        }
    }

    // ---- ReadableMap helpers -------------------------------------------------------------
    private fun ReadableMap?.typeOf(k: String): ReadableType? =
        if (this != null && hasKey(k) && !isNull(k)) getType(k) else null
    private fun ReadableMap?.str(k: String): String? = if (typeOf(k) == ReadableType.String) this?.getString(k) else null
    private fun ReadableMap?.bool(k: String): Boolean? = if (typeOf(k) == ReadableType.Boolean) this?.getBoolean(k) else null
    private fun ReadableMap?.num(k: String): Double? = if (typeOf(k) == ReadableType.Number) this?.getDouble(k) else null
    private fun ReadableMap?.finite(k: String): Double? = num(k)?.takeIf { it.isFinite() }

    private fun ReadableMap?.strList(k: String): List<String>? {
        if (typeOf(k) != ReadableType.Array) return null
        val a = this?.getArray(k) ?: return null
        return (0 until a.size()).mapNotNull { i -> if (a.getType(i) == ReadableType.String) a.getString(i) else null }
    }

    private fun ReadableMap?.strMap(k: String): Map<String, String>? {
        if (typeOf(k) != ReadableType.Map) return null
        val m = this?.getMap(k) ?: return null
        val out = LinkedHashMap<String, String>()
        val it = m.keySetIterator()
        while (it.hasNextKey()) {
            val key = it.nextKey()
            if (!m.isNull(key) && m.getType(key) == ReadableType.String) m.getString(key)?.let { v -> out[key] = v }
        }
        return out
    }
}

// ---- JSON → bridge values ------------------------------------------------------------------
internal fun toWritableMap(o: JSONObject): WritableMap {
    val map = Arguments.createMap()
    val keys = o.keys()
    while (keys.hasNext()) {
        val k = keys.next()
        when (val v = o.opt(k)) {
            null, JSONObject.NULL -> map.putNull(k)
            is JSONObject -> map.putMap(k, toWritableMap(v))
            is JSONArray -> map.putArray(k, toWritableArray(v))
            is Boolean -> map.putBoolean(k, v)
            is Int -> map.putInt(k, v)
            is Number -> map.putDouble(k, v.toDouble())
            else -> map.putString(k, v.toString())
        }
    }
    return map
}

internal fun toWritableArray(a: JSONArray): WritableArray {
    val arr = Arguments.createArray()
    for (i in 0 until a.length()) {
        when (val v = a.opt(i)) {
            null, JSONObject.NULL -> arr.pushNull()
            is JSONObject -> arr.pushMap(toWritableMap(v))
            is JSONArray -> arr.pushArray(toWritableArray(v))
            is Boolean -> arr.pushBoolean(v)
            is Int -> arr.pushInt(v)
            is Number -> arr.pushDouble(v.toDouble())
            else -> arr.pushString(v.toString())
        }
    }
    return arr
}
