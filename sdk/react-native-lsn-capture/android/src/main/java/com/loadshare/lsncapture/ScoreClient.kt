package com.loadshare.lsncapture

import android.util.Base64
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.InterruptedIOException
import java.net.URI
import java.net.UnknownServiceException
import java.util.concurrent.TimeUnit

/** A failed score call; [code] is one of the SDK's LsnErrorCode strings. */
class ScoreException(val code: String, message: String) : Exception(message)

/**
 * The validated inputs of `POST {apiBase}/v1/checkpoint/score`.
 * Mirrors LsnCaptureConfig's checks in the Flutter SDK (api.dart), plus the
 * limits below. Keep in sync with `validateScoreOptions` in src/scores.ts.
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
    /** Never prints the selfie / header values (they can hold tokens or images). */
    override fun toString(): String =
        "ScoreRequest(apiBase=$apiBase, checks=$checks, faceCheckPath=$faceCheckPath, faceCheckS3=${faceCheckS3 != null}, " +
            "sourceSelfieS3=${sourceSelfieS3 != null}, sourceSelfieB64=${sourceSelfieB64?.length}, timeoutMs=$timeoutMs, headers=${headers.keys})"

    companion object {
        val ALL_CHECKS = listOf("face_match", "dress_color", "logo")
        const val DEFAULT_TIMEOUT_MS = 40_000L
        const val MAX_TIMEOUT_MS = 600_000L

        /** The server's per-image limit (zepiris/schemas/face.py MAX_IMAGE_SIZE_BYTES, 5 MiB). */
        const val MAX_IMAGE_BYTES = 5L * 1024 * 1024
        /** Base64 of a MAX_IMAGE_BYTES image, plus room for a `data:` prefix. */
        const val MAX_SELFIE_B64_CHARS = (MAX_IMAGE_BYTES + 2) / 3 * 4 + 64

        /** Set by the SDK / OkHttp; a host value would corrupt the request, so it is dropped. */
        private val RESERVED_HEADERS = setOf("content-type", "content-length", "transfer-encoding", "host", "connection")
        private val TOKEN = Regex("^[!#$%&'*+.^_`|~0-9A-Za-z-]+$") // RFC 9110 field-name

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
            if (uri.rawQuery != null || uri.rawFragment != null) {
                bad("apiBase must not have a query or fragment")
            }

            val list = (checks ?: ALL_CHECKS).distinct()
            if (list.isEmpty() || list.any { it !in ALL_CHECKS }) {
                bad("checks must be a non-empty subset of $ALL_CHECKS")
            }

            val s3Selfie = sourceSelfieS3?.takeIf { it.isNotEmpty() }
            val b64Selfie = sourceSelfieB64?.takeIf { it.isNotEmpty() }
            if ("face_match" in list && s3Selfie == null && b64Selfie == null) {
                bad("face_match needs sourceSelfieS3 or sourceSelfieB64")
            }
            if (b64Selfie != null && b64Selfie.length > MAX_SELFIE_B64_CHARS) {
                bad("sourceSelfieB64 is larger than the server's 5 MB image limit")
            }

            val path = faceCheckPath?.takeIf { it.isNotEmpty() }?.removePrefix("file://")
            val s3 = faceCheckS3?.takeIf { it.isNotEmpty() }
            if ((path == null) == (s3 == null)) {
                bad("pass exactly one of faceCheckPath or faceCheckS3")
            }

            val timeout = timeoutMs ?: DEFAULT_TIMEOUT_MS
            if (timeout <= 0 || timeout > MAX_TIMEOUT_MS) bad("timeoutMs must be > 0 and <= $MAX_TIMEOUT_MS")

            // OkHttp would throw on these at call time (and its message can echo the value).
            // Headers the SDK owns (Content-Type, Content-Length, Host, ...) are dropped.
            val hdrs = LinkedHashMap<String, String>()
            for ((name, value) in headers ?: emptyMap()) {
                if (!TOKEN.matches(name)) bad("headers: invalid header name")
                if (value.any { it != '\t' && (it < ' ' || it > '~') }) {
                    bad("headers: the value of $name has a control (CR/LF) or non-ASCII character")
                }
                if (name.lowercase() !in RESERVED_HEADERS) hdrs[name] = value
            }

            return ScoreRequest(base, list, path, s3, s3Selfie, b64Selfie, timeout, hdrs)
        }
    }
}

/**
 * `POST {apiBase}/v1/checkpoint/score` with OkHttp (bundled with React Native).
 * Blocking: call it off the main thread. The photo is read and base64-encoded
 * here, so the multi-MB JPEG never crosses the JS bridge.
 *
 * Uses only OkHttp APIs present in both 4.9.x (RN's default) and 5.x.
 */
object ScoreClient {
    /** Tracks package.json's version (BuildConfig.SDK_VERSION is generated from it). */
    val USER_AGENT = "LSNCaptureSDK-RN/${BuildConfig.SDK_VERSION}"
    /** The score reply is a few KB of JSON; anything far bigger is not our server. */
    const val MAX_RESPONSE_BYTES = 1_000_000
    private const val MAX_CONNECT_TIMEOUT_MS = 15_000L
    private const val MAX_MESSAGE_CHARS = 300

    private val JSON by lazy { "application/json".toMediaType() }
    /** One client per process (shared connection pool); per-call timeouts via newBuilder(). */
    private val base: OkHttpClient by lazy { OkHttpClient() }

    private fun androidBase64(bytes: ByteArray): ByteArray = Base64.encode(bytes, Base64.NO_WRAP)

    /** Returns the server's JSON as sent; throws [ScoreException] otherwise. */
    fun score(req: ScoreRequest): JSONObject = score(req, base, ::androidBase64)

    internal fun score(req: ScoreRequest, client: OkHttpClient, encode: (ByteArray) -> ByteArray): JSONObject {
        val body = buildBody(req, encode)
        val call = client.newBuilder()
            .callTimeout(req.timeoutMs, TimeUnit.MILLISECONDS)
            .connectTimeout(minOf(req.timeoutMs, MAX_CONNECT_TIMEOUT_MS), TimeUnit.MILLISECONDS)
            .readTimeout(req.timeoutMs, TimeUnit.MILLISECONDS)
            .writeTimeout(req.timeoutMs, TimeUnit.MILLISECONDS)
            .followSslRedirects(false) // never follow an https -> http redirect with the photo
            .build()
        val request = try {
            val builder = Request.Builder()
                .url("${req.apiBase}/v1/checkpoint/score")
                .header("User-Agent", USER_AGENT)
            req.headers.forEach { (k, v) -> builder.header(k, v) }
            builder.post(body.toRequestBody(JSON, 0, body.size)).build()
        } catch (e: IllegalArgumentException) { // no e.message: it can echo a header value
            throw ScoreException("invalid_config", "apiBase or headers are not valid for an HTTP request")
        }

        val code: Int
        val text: String
        try {
            call.newCall(request).execute().use { r ->
                code = r.code
                text = readCapped(r.body?.byteStream())
            }
        } catch (e: InterruptedIOException) { // SocketTimeoutException and OkHttp's call timeout
            throw ScoreException("timeout", "The server took too long. Check the connection and try again.")
        } catch (e: UnknownServiceException) { // cleartext blocked by the app's network security config
            throw ScoreException("invalid_config", "apiBase is http:// but the app does not allow cleartext traffic to that host; use https")
        } catch (e: IOException) {
            throw ScoreException("network", "Could not reach the server. Check the connection and try again.")
        }
        return parseResponse(code, text)
    }

    /**
     * The request JSON as UTF-8 bytes. The photo's base64 is spliced in as raw
     * ASCII (it needs no JSON escaping), so the image is held as bytes + base64
     * + body (~3.7x the file) rather than also as escaped strings.
     */
    internal fun buildBody(req: ScoreRequest, encode: (ByteArray) -> ByteArray): ByteArray {
        val rest = JSONObject()
        if (req.faceCheckPath == null) rest.put("face_check_s3", req.faceCheckS3)
        rest.put("checks", JSONArray(req.checks))
        req.sourceSelfieS3?.let { rest.put("source_selfie_s3", it) }
        req.sourceSelfieB64?.let { rest.put("source_selfie_b64", it) }
        val restJson = rest.toString() // always "{...}" with at least "checks"
        if (req.faceCheckPath == null) return restJson.toByteArray(Charsets.UTF_8)

        val b64 = encode(readImage(req.faceCheckPath))
        val head = "{\"face_check_b64\":\"".toByteArray(Charsets.US_ASCII)
        val tail = ("\"," + restJson.substring(1)).toByteArray(Charsets.UTF_8)
        val out = ByteArray(head.size + b64.size + tail.size)
        System.arraycopy(head, 0, out, 0, head.size)
        System.arraycopy(b64, 0, out, head.size, b64.size)
        System.arraycopy(tail, 0, out, head.size + b64.size, tail.size)
        return out
    }

    /** The JPEG at [path], or `invalid_config` if it is missing, empty, unreadable or over the server's limit. */
    internal fun readImage(path: String): ByteArray {
        fun bad(msg: String): Nothing = throw ScoreException("invalid_config", msg)
        val file = File(path)
        if (!file.isFile) bad("faceCheckPath not found: $path")
        val len = file.length()
        if (len <= 0L) bad("faceCheckPath is empty: $path")
        if (len > ScoreRequest.MAX_IMAGE_BYTES) bad("faceCheckPath is $len bytes; the server takes at most ${ScoreRequest.MAX_IMAGE_BYTES}")
        val bytes = try { file.readBytes() } catch (e: IOException) { bad("faceCheckPath unreadable: ${e.message}") }
        if (bytes.isEmpty() || bytes.size > ScoreRequest.MAX_IMAGE_BYTES) bad("faceCheckPath changed while reading: $path")
        return bytes
    }

    /** The reply body as UTF-8, at most [MAX_RESPONSE_BYTES] (code `server` beyond that). */
    internal fun readCapped(input: InputStream?): String {
        if (input == null) return ""
        val out = ByteArrayOutputStream()
        val buf = ByteArray(8192)
        var total = 0
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            total += n
            if (total > MAX_RESPONSE_BYTES) throw ScoreException("server", "The server sent an unexpectedly large reply.")
            out.write(buf, 0, n)
        }
        return String(out.toByteArray(), Charsets.UTF_8)
    }

    /** Status + body text → the score JSON, or a [ScoreException] (code `server`). */
    fun parseResponse(code: Int, text: String): JSONObject {
        val parsed: Any? = try { JSONTokener(text).nextValue() } catch (e: Exception) { null }
        if (code !in 200..299) throw ScoreException("server", errorMessage(code, parsed))
        if (parsed !is JSONObject) throw ScoreException("server", "The server sent an unexpected reply ($code).")
        if (parsed.optJSONObject("scores") == null) throw ScoreException("server", "The server sent no scores.")
        return parsed
    }

    /** FastAPI's `detail`: a string, `{message}`, or a validation list of `{msg}`. Capped at 300 chars. */
    fun errorMessage(code: Int, parsed: Any?): String {
        val detail = (parsed as? JSONObject)?.opt("detail")
        val msg = when (detail) {
            is String -> detail.ifEmpty { null }
            is JSONObject -> detail.optString("message").ifEmpty { detail.toString() }
            is JSONArray -> if (detail.length() > 0) {
                (detail.opt(0) as? JSONObject)?.optString("msg")?.ifEmpty { null } ?: "invalid request"
            } else null
            else -> null
        } ?: "HTTP $code"
        return if (msg.length > MAX_MESSAGE_CHARS) msg.take(MAX_MESSAGE_CHARS) + "…" else msg
    }
}
