package com.loadshare.lsncapture

import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File
import java.util.Base64

class ScoreClientTest {
    private fun ok(
        apiBase: String? = "https://3-108-193-187.sslip.io//",
        checks: List<String>? = null,
        path: String? = "/data/x.jpg",
        s3: String? = null,
        selfieS3: String? = "https://b.s3.amazonaws.com/r.jpg",
        selfieB64: String? = null,
        timeoutMs: Long? = null,
        headers: Map<String, String>? = null,
    ) = ScoreRequest.validate(apiBase, checks, path, s3, selfieS3, selfieB64, timeoutMs, headers)

    private fun invalid(block: () -> Unit) { expectCode("invalid_config", block) }

    private fun expectCode(code: String, block: () -> Unit): ScoreException {
        try { block() } catch (e: ScoreException) { assertEquals(e.message, code, e.code); return e }
        fail("expected $code"); throw AssertionError()
    }

    private val jvmBase64: (ByteArray) -> ByteArray = { Base64.getEncoder().encode(it) }
    private val tmp = mutableListOf<File>()
    private var server: MockWebServer? = null

    @After fun cleanUp() {
        tmp.forEach { it.delete() }
        server?.shutdown()
    }

    private fun jpeg(size: Int): File = File.createTempFile("lsn_capture_test", ".jpg").also {
        tmp += it
        it.writeBytes(ByteArray(size) { i -> (i % 251).toByte() })
    }

    private fun startServer(vararg responses: MockResponse): MockWebServer =
        MockWebServer().also { s -> responses.forEach { s.enqueue(it) }; s.start(); server = s }

    private fun reqFor(s: MockWebServer, path: String? = null, s3: String? = "https://b/x.jpg", timeoutMs: Long? = 5_000, headers: Map<String, String>? = null) =
        ok(apiBase = s.url("/").toString(), path = path, s3 = s3, timeoutMs = timeoutMs, headers = headers)

    @Test fun defaults() {
        val r = ok()
        assertEquals("https://3-108-193-187.sslip.io", r.apiBase)
        assertEquals(listOf("face_match", "dress_color", "logo"), r.checks)
        assertEquals(40_000L, r.timeoutMs)
        assertNull(r.faceCheckS3)
    }
    @Test fun fileUriPrefixStripped() = assertEquals("/data/x.jpg", ok(path = "file:///data/x.jpg").faceCheckPath)
    @Test fun badScheme() = invalid { ok(apiBase = "ftp://x.io") }
    @Test fun noHost() = invalid { ok(apiBase = "https://") }
    @Test fun emptyChecks() = invalid { ok(checks = emptyList()) }
    @Test fun unknownCheck() = invalid { ok(checks = listOf("face_match", "hat")) }
    @Test fun faceMatchNeedsSelfie() = invalid { ok(selfieS3 = null) }
    @Test fun noSelfieNeededWithoutFaceMatch() = assertEquals(listOf("logo"), ok(checks = listOf("logo"), selfieS3 = null).checks)
    @Test fun b64SelfieIsEnough() = assertEquals("abc", ok(selfieS3 = "", selfieB64 = "abc").sourceSelfieB64)
    @Test fun bothPhotoSources() = invalid { ok(s3 = "https://b/x.jpg") }
    @Test fun noPhotoSource() = invalid { ok(path = null) }

    @Test fun detailString() = assertEquals("bad image", ScoreClient.errorMessage(422, JSONObject("""{"detail":"bad image"}""")))
    @Test fun detailMessage() = assertEquals("too big", ScoreClient.errorMessage(413, JSONObject("""{"detail":{"message":"too big"}}""")))
    @Test fun detailList() = assertEquals("field required", ScoreClient.errorMessage(422, JSONObject("""{"detail":[{"msg":"field required"}]}""")))
    @Test fun noDetail() = assertEquals("HTTP 502", ScoreClient.errorMessage(502, null))

    @Test fun non2xxIsServer() = serverError { ScoreClient.parseResponse(500, """{"detail":"boom"}""") }
    @Test fun htmlIsServer() = serverError { ScoreClient.parseResponse(200, "<html>proxy</html>") }
    @Test fun missingScoresIsServer() = serverError { ScoreClient.parseResponse(200, """{"requestId":"r"}""") }
    @Test fun scoresPass() = assertEquals("r1", ScoreClient.parseResponse(200, """{"requestId":"r1","scores":{}}""").getString("requestId"))

    private fun serverError(block: () -> Unit) {
        expectCode("server", block)
    }

    // ---- validation: limits and headers ------------------------------------------------------
    @Test fun apiBaseWithQuery() = invalid { ok(apiBase = "https://x.io/?token=1") }
    @Test fun apiBaseWithFragment() = invalid { ok(apiBase = "https://x.io/#a") }
    @Test fun apiBaseKeepsPath() = assertEquals("https://x.io/api", ok(apiBase = " https://x.io/api/ ").apiBase)
    @Test fun apiBaseWithSpace() = invalid { ok(apiBase = "https://x io") }
    @Test fun duplicateChecksCollapsed() = assertEquals(listOf("logo"), ok(checks = listOf("logo", "logo"), selfieS3 = null).checks)
    @Test fun zeroTimeout() = invalid { ok(timeoutMs = 0) }
    @Test fun negativeTimeout() = invalid { ok(timeoutMs = -1) }
    @Test fun hugeTimeout() = invalid { ok(timeoutMs = ScoreRequest.MAX_TIMEOUT_MS + 1) }
    @Test fun maxTimeoutOk() = assertEquals(ScoreRequest.MAX_TIMEOUT_MS, ok(timeoutMs = ScoreRequest.MAX_TIMEOUT_MS).timeoutMs)
    @Test fun selfieB64TooLarge() = invalid {
        ok(selfieS3 = null, selfieB64 = "A".repeat((ScoreRequest.MAX_SELFIE_B64_CHARS + 1).toInt()))
    }
    @Test fun headerValueWithCrLf() {
        val e = expectCode("invalid_config") { ok(headers = mapOf("X-Token" to "abc\r\nX-Evil: 1")) }
        assertFalse("message must not echo the value", e.message!!.contains("abc"))
    }
    @Test fun headerValueWithNul() = invalid { ok(headers = mapOf("X-Token" to "a\u0000b")) }
    @Test fun headerValueNonAscii() = invalid { ok(headers = mapOf("X-Token" to "tökén")) }
    @Test fun headerNameWithSpace() = invalid { ok(headers = mapOf("X Token" to "v")) }
    @Test fun headerNameWithNewline() = invalid { ok(headers = mapOf("X-Token\n" to "v")) }
    @Test fun headerNameEmpty() = invalid { ok(headers = mapOf("" to "v")) }
    @Test fun headerTabAllowed() = assertEquals("a\tb", ok(headers = mapOf("X-Token" to "a\tb")).headers["X-Token"])
    @Test fun reservedHeadersDropped() = assertEquals(
        mapOf("Authorization" to "Bearer t"),
        ok(headers = mapOf("Content-Type" to "text/plain", "content-length" to "1", "Host" to "evil", "Authorization" to "Bearer t")).headers,
    )
    @Test fun toStringHidesSecrets() {
        val r = ok(selfieS3 = null, selfieB64 = "SECRETB64", headers = mapOf("Authorization" to "Bearer SECRET"))
        assertFalse(r.toString().contains("SECRET"))
    }
    @Test fun longDetailTruncated() {
        val msg = ScoreClient.errorMessage(500, JSONObject().put("detail", "x".repeat(5000)))
        assertTrue(msg.length <= 301)
    }
    @Test fun emptyBodyNon2xx() = assertEquals("HTTP 503", expectCode("server") { ScoreClient.parseResponse(503, "") }.message)
    @Test fun emptyBody2xx() = serverError { ScoreClient.parseResponse(200, "") }
    @Test fun jsonArrayReply() = serverError { ScoreClient.parseResponse(200, "[1,2]") }

    // ---- photo file ---------------------------------------------------------------------------
    @Test fun missingFile() = invalid { ScoreClient.readImage("/nope/lsn_capture_missing.jpg") }
    @Test fun emptyFile() = invalid { ScoreClient.readImage(jpeg(0).path) }
    @Test fun tooLargeFile() = invalid { ScoreClient.readImage(jpeg(ScoreRequest.MAX_IMAGE_BYTES.toInt() + 1).path) }
    @Test fun maxSizeFileOk() = assertEquals(ScoreRequest.MAX_IMAGE_BYTES.toInt(), ScoreClient.readImage(jpeg(ScoreRequest.MAX_IMAGE_BYTES.toInt()).path).size)

    @Test fun bodyWithFileIsValidJson() {
        val f = jpeg(1000)
        val body = ScoreClient.buildBody(ok(path = "file://" + f.path, selfieS3 = "https://b/r.jpg"), jvmBase64)
        val json = JSONObject(String(body, Charsets.UTF_8))
        assertEquals(f.readBytes().toList(), Base64.getDecoder().decode(json.getString("face_check_b64")).toList())
        assertEquals(listOf("face_match", "dress_color", "logo"), (0 until 3).map { json.getJSONArray("checks").getString(it) })
        assertEquals("https://b/r.jpg", json.getString("source_selfie_s3"))
        assertFalse(json.has("face_check_s3"))
    }

    @Test fun bodyWithS3() {
        val body = ScoreClient.buildBody(ok(path = null, s3 = "https://b/x.jpg", checks = listOf("logo"), selfieS3 = null), jvmBase64)
        val json = JSONObject(String(body, Charsets.UTF_8))
        assertEquals("https://b/x.jpg", json.getString("face_check_s3"))
        assertFalse(json.has("face_check_b64"))
        assertFalse(json.has("source_selfie_s3"))
    }

    // ---- reply size cap -------------------------------------------------------------------------
    @Test fun readCappedSmall() = assertEquals("{\"a\":\"é\"}", ScoreClient.readCapped(ByteArrayInputStream("{\"a\":\"é\"}".toByteArray())))
    @Test fun readCappedNull() = assertEquals("", ScoreClient.readCapped(null))
    @Test fun readCappedTooLarge() = serverError {
        ScoreClient.readCapped(ByteArrayInputStream(ByteArray(ScoreClient.MAX_RESPONSE_BYTES + 1)))
    }

    // ---- end to end against a local server (OkHttp 4.9.2, RN 0.77's version) ----------------------
    private val scoresJson = """{"requestId":"r1","scores":{"face_match":{"similarity":0.8}}}"""

    @Test fun e2eSendsPhotoHeadersAndUserAgent() {
        val s = startServer(MockResponse().setBody(scoresJson))
        val f = jpeg(2048)
        val out = ScoreClient.score(reqFor(s, path = f.path, s3 = null, headers = mapOf("X-Gateway" to "tok")), OkHttpClient(), jvmBase64)
        assertEquals("r1", out.getString("requestId"))
        val r = s.takeRequest()
        assertEquals("POST", r.method)
        assertEquals("/v1/checkpoint/score", r.path)
        assertEquals("tok", r.getHeader("X-Gateway"))
        assertTrue(r.getHeader("Content-Type")!!.startsWith("application/json"))
        assertEquals("LSNCaptureSDK-RN/" + BuildConfig.SDK_VERSION, r.getHeader("User-Agent"))
        assertTrue(BuildConfig.SDK_VERSION.matches(Regex("\\d+\\.\\d+\\.\\d+.*")))
        val sent = JSONObject(r.body.readUtf8())
        assertEquals(f.readBytes().toList(), Base64.getDecoder().decode(sent.getString("face_check_b64")).toList())
    }

    @Test fun e2eServerError() {
        val s = startServer(MockResponse().setResponseCode(422).setBody("""{"detail":"no face"}"""))
        assertEquals("no face", expectCode("server") { ScoreClient.score(reqFor(s), OkHttpClient(), jvmBase64) }.message)
    }

    @Test fun e2eHtmlReply() {
        val s = startServer(MockResponse().setResponseCode(502).setBody("<html>bad gateway</html>"))
        assertEquals("HTTP 502", expectCode("server") { ScoreClient.score(reqFor(s), OkHttpClient(), jvmBase64) }.message)
    }

    @Test fun e2eHugeReply() {
        val s = startServer(MockResponse().setBody("x".repeat(ScoreClient.MAX_RESPONSE_BYTES + 10)))
        expectCode("server") { ScoreClient.score(reqFor(s), OkHttpClient(), jvmBase64) }
    }

    @Test fun e2eTimeout() {
        val s = startServer(MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE))
        expectCode("timeout") { ScoreClient.score(reqFor(s, timeoutMs = 300), OkHttpClient(), jvmBase64) }
    }

    @Test fun e2eConnectionRefused() {
        val s = startServer()
        val req = reqFor(s)
        s.shutdown(); server = null
        expectCode("network") { ScoreClient.score(req, OkHttpClient(), jvmBase64) }
    }

    @Test fun e2eMissingFileFailsBeforeNetwork() {
        val s = startServer(MockResponse().setBody(scoresJson))
        invalid { ScoreClient.score(reqFor(s, path = "/nope/x.jpg", s3 = null), OkHttpClient(), jvmBase64) }
        assertEquals(0, s.requestCount)
    }
}
