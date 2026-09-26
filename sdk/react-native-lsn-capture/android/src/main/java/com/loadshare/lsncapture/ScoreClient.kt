package com.loadshare.lsncapture

import android.util.Base64
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.File
import java.io.IOException
import java.io.InterruptedIOException
import java.net.URI
import java.util.concurrent.TimeUnit

/** A failed score call; [code] is one of the SDK's LsnErrorCode strings. */
class ScoreException(val code: String, message: String) : Exception(message)

/**
 * The validated inputs of `POST {apiBase}/v1/checkpoint/score`.
 * Mirrors LsnCaptureConfig's checks in the Flutter SDK (api.dart).
 */
data class ScoreRequest(
    val apiBase: String,
    val checks: List<String>,
    val faceCheckPath: String?,
    val faceCheckS3: String?,
    val sourceSelfieS3: String?,
    val sourceSelfieB64: String?,
    val timeoutMs: Long,
    val headers: Map<String, String>,
) {
    companion object {
        val ALL_CHECKS = listOf("face_match", "dress_color", "logo")
        const val DEFAULT_TIMEOUT_MS = 40_000L

        /** Throws [ScoreException] with code `invalid_config` when the inputs are unusable. */
        fun validate(
            apiBase: String?,
            checks: List<String>?,
            faceCheckPath: String?,
            faceCheckS3: String?,
            sourceSelfieS3: String?,
            sourceSelfieB64: String?,
            timeoutMs: Long?,
            headers: Map<String, String>?,
        ): ScoreRequest {
            fun bad(msg: String): Nothing = throw ScoreException("invalid_config", msg)

            val base = (apiBase ?: "").trim().replace(Regex("/+$"), "")
            val uri = try { URI(base) } catch (e: Exception) { null }
            val scheme = uri?.scheme?.lowercase()
            if (uri == null || (scheme != "http" && scheme != "https") || uri.host.isNullOrEmpty()) {
                bad("apiBase must be an http(s) URL")
            }

            val list = checks ?: ALL_CHECKS
            if (list.isEmpty() || list.any { it !in ALL_CHECKS }) {
                bad("checks must be a non-empty subset of $ALL_CHECKS")
            }

            val s3Selfie = sourceSelfieS3?.takeIf { it.isNotEmpty() }
            val b64Selfie = sourceSelfieB64?.takeIf { it.isNotEmpty() }
            if ("face_match" in list && s3Selfie == null && b64Selfie == null) {
                bad("face_match needs sourceSelfieS3 or sourceSelfieB64")
            }

            val path = faceCheckPath?.takeIf { it.isNotEmpty() }?.removePrefix("file://")
            val s3 = faceCheckS3?.takeIf { it.isNotEmpty() }
            if ((path == null) == (s3 == null)) {
                bad("pass exactly one of faceCheckPath or faceCheckS3")
            }

            val timeout = timeoutMs ?: DEFAULT_TIMEOUT_MS
            if (timeout <= 0) bad("timeoutMs must be > 0")

            return ScoreRequest(base, list, path, s3, s3Selfie, b64Selfie, timeout, headers ?: emptyMap())
        }
    }
}

/**
 * `POST {apiBase}/v1/checkpoint/score` with OkHttp (bundled with React Native).
 * Blocking: call it off the main thread. The photo is read and base64-encoded
 * here, so the multi-MB JPEG never crosses the JS bridge.
 */
object ScoreClient {
    private const val USER_AGENT = "LSNCaptureSDK-RN/1.0"
    private val JSON by lazy { "application/json".toMediaType() }
    private val base: OkHttpClient by lazy { OkHttpClient() }

    /** Returns the server's JSON as sent; throws [ScoreException] otherwise. */
    fun score(req: ScoreRequest): JSONObject {
        val body = buildBody(req)
        val client = base.newBuilder()
            .callTimeout(req.timeoutMs, TimeUnit.MILLISECONDS)
            .connectTimeout(req.timeoutMs, TimeUnit.MILLISECONDS)
            .readTimeout(req.timeoutMs, TimeUnit.MILLISECONDS)
            .writeTimeout(req.timeoutMs, TimeUnit.MILLISECONDS)
            .build()
        val builder = Request.Builder()
            .url("${req.apiBase}/v1/checkpoint/score")
            .header("Content-Type", "application/json")
            .header("User-Agent", USER_AGENT)
        req.headers.forEach { (k, v) -> builder.header(k, v) }
        val request = builder.post(body.toString().toRequestBody(JSON)).build()

        val code: Int
        val text: String
        try {
            client.newCall(request).execute().use { r ->
                code = r.code
                text = r.body?.string() ?: ""
            }
        } catch (e: InterruptedIOException) { // SocketTimeoutException and OkHttp's call timeout
            throw ScoreException("timeout", "The server took too long. Check the connection and try again.")
        } catch (e: IOException) {
            throw ScoreException("network", "Could not reach the server. Check the connection and try again.")
        }
        return parseResponse(code, text)
    }

    fun buildBody(req: ScoreRequest): JSONObject {
        val body = JSONObject()
        if (req.faceCheckPath != null) {
            val file = File(req.faceCheckPath)
            if (!file.isFile) throw ScoreException("invalid_config", "faceCheckPath not found: ${req.faceCheckPath}")
            val bytes = try { file.readBytes() } catch (e: IOException) {
                throw ScoreException("invalid_config", "faceCheckPath unreadable: ${e.message}")
            }
            body.put("face_check_b64", Base64.encodeToString(bytes, Base64.NO_WRAP))
        } else {
            body.put("face_check_s3", req.faceCheckS3)
        }
        body.put("checks", JSONArray(req.checks))
        req.sourceSelfieS3?.let { body.put("source_selfie_s3", it) }
        req.sourceSelfieB64?.let { body.put("source_selfie_b64", it) }
        return body
    }

    /** Status + body text → the score JSON, or a [ScoreException] (code `server`). */
    fun parseResponse(code: Int, text: String): JSONObject {
        val parsed: Any? = try { JSONTokener(text).nextValue() } catch (e: Exception) { null }
        if (code !in 200..299) throw ScoreException("server", errorMessage(code, parsed))
        if (parsed !is JSONObject) throw ScoreException("server", "The server sent an unexpected reply ($code).")
        if (parsed.optJSONObject("scores") == null) throw ScoreException("server", "The server sent no scores.")
        return parsed
    }

    /** FastAPI's `detail`: a string, `{message}`, or a validation list of `{msg}`. */
    fun errorMessage(code: Int, parsed: Any?): String {
        val detail = (parsed as? JSONObject)?.opt("detail")
        return when (detail) {
            is String -> detail.ifEmpty { null }
            is JSONObject -> detail.optString("message").ifEmpty { detail.toString() }
            is JSONArray -> if (detail.length() > 0) {
                (detail.opt(0) as? JSONObject)?.optString("msg")?.ifEmpty { null } ?: "invalid request"
            } else null
            else -> null
        } ?: "HTTP $code"
    }
}
