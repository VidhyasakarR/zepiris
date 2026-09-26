package com.loadshare.lsncapture

import android.app.Activity
import android.content.Intent
import com.facebook.react.bridge.ActivityEventListener
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.ReadableType
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import org.json.JSONArray
import org.json.JSONObject
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
 */
class LsnCaptureModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext), ActivityEventListener {

    companion object {
        const val NAME = "LsnCapture"
        private const val REQUEST = 0x15C4 // same request code as the Flutter plugin
    }

    private val lock = Any()
    private var pending: Promise? = null
    private val executor: ExecutorService = Executors.newCachedThreadPool()

    init {
        reactContext.addActivityEventListener(this)
    }

    override fun getName(): String = NAME

    override fun invalidate() {
        reactContext.removeActivityEventListener(this)
        executor.shutdown()
        super.invalidate()
    }

    // ---- warmUp ------------------------------------------------------------------------
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
        val activity = reactContext.currentActivity
            ?: return promise.reject("no_activity", "no foreground activity")
        synchronized(lock) {
            if (pending != null) return promise.reject("busy", "a capture is already running")
            pending = promise
        }
        try {
            activity.startActivityForResult(
                Intent(activity, CaptureActivity::class.java)
                    .putExtra(CaptureActivity.EXTRA_CHALLENGE, opts.str("challenge") ?: "blink")
                    .putExtra(CaptureActivity.EXTRA_LIGHT, opts.bool("light") ?: false)
                    .putExtra(CaptureActivity.EXTRA_MAX_SIDE, opts.num("maxSide")?.toInt() ?: 2592)
                    .putExtra(CaptureActivity.EXTRA_JPEG_QUALITY, opts.num("jpegQuality")?.toInt() ?: 92)
                    .putExtra(CaptureActivity.EXTRA_BRIGHTNESS, (opts.num("brightness") ?: 0.0).toFloat()),
                REQUEST,
            )
        } catch (e: Exception) {
            synchronized(lock) { pending = null } // never leave the module stuck on "busy"
            promise.reject("capture_failed", e.message ?: "could not open the capture screen", e)
        }
    }

    override fun onActivityResult(activity: Activity, requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode != REQUEST) return
        val p = synchronized(lock) { pending.also { pending = null } } ?: return
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

    override fun onNewIntent(intent: Intent) {}

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
                timeoutMs = opts.num("timeoutMs")?.toLong(),
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
                } catch (e: Exception) {
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
