package com.loadshare.lsn_capture_sdk

import android.app.Activity
import android.content.Intent
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.PluginRegistry

/**
 * Dart bridge: `warmUp` (load the face model + CameraX ahead of time) and
 * `capture` (run [CaptureActivity], answer with the photo path and stats).
 */
class LsnCaptureSdkPlugin : FlutterPlugin, MethodChannel.MethodCallHandler, ActivityAware,
    PluginRegistry.ActivityResultListener {
    private lateinit var channel: MethodChannel
    private var binding: ActivityPluginBinding? = null
    private var pending: MethodChannel.Result? = null
    private var appContext: android.content.Context? = null

    companion object { private const val REQUEST = 0x15C4 }

    override fun onAttachedToEngine(b: FlutterPlugin.FlutterPluginBinding) {
        appContext = b.applicationContext
        channel = MethodChannel(b.binaryMessenger, "lsn_capture_sdk")
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(b: FlutterPlugin.FlutterPluginBinding) = channel.setMethodCallHandler(null)

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "warmUp" -> {
                val ctx = appContext ?: return result.error("no_context", null, null)
                FaceEngine.warmUp(ctx) { r -> android.os.Handler(android.os.Looper.getMainLooper()).post { result.success(r) } }
            }
            "capture" -> {
                val activity = binding?.activity ?: return result.error("no_activity", "no foreground activity", null)
                if (pending != null) return result.error("busy", "a capture is already running", null)
                pending = result
                try {
                    activity.startActivityForResult(
                        Intent(activity, CaptureActivity::class.java)
                            .putExtra(CaptureActivity.EXTRA_CHALLENGE, call.argument<String>("challenge") ?: "blink")
                            .putExtra("debug", call.argument<Boolean>("debug") ?: false)
                        .putExtra(CaptureActivity.EXTRA_LIGHT, call.argument<Boolean>("light") ?: false)
                        .putExtra(CaptureActivity.EXTRA_MAX_SIDE, call.argument<Int>("maxSide") ?: 2592)
                        .putExtra(CaptureActivity.EXTRA_JPEG_QUALITY, call.argument<Int>("jpegQuality") ?: 92)
                        .putExtra(CaptureActivity.EXTRA_BRIGHTNESS, (call.argument<Number>("brightness") ?: 0).toFloat()),
                        REQUEST,
                    )
                } catch (e: Exception) {
                    pending = null // never leave the plugin stuck on "busy"
                    result.error("capture_failed", e.message, null)
                }
            }
            else -> result.notImplemented()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode != REQUEST) return false
        val r = pending ?: return true
        pending = null
        when (resultCode) {
            Activity.RESULT_OK -> r.success(mapOf(
                "path" to data?.getStringExtra(CaptureActivity.RESULT_PATH),
                "stats" to data?.getStringExtra(CaptureActivity.RESULT_JSON),
            ))
            Activity.RESULT_CANCELED -> r.success(null)
            else -> r.error(data?.getStringExtra("error") ?: "capture_failed", data?.getStringExtra("error"), null)
        }
        return true
    }

    override fun onAttachedToActivity(b: ActivityPluginBinding) {
        binding = b
        b.addActivityResultListener(this)
    }

    override fun onDetachedFromActivity() {
        binding?.removeActivityResultListener(this)
        binding = null
    }

    override fun onReattachedToActivityForConfigChanges(b: ActivityPluginBinding) = onAttachedToActivity(b)
    override fun onDetachedFromActivityForConfigChanges() = onDetachedFromActivity()
}
