package com.loadshare.lsncapture

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Test

class ScoreClientTest {
    private fun ok(
        apiBase: String? = "https://3-108-193-187.sslip.io//",
        checks: List<String>? = null,
        path: String? = "/data/x.jpg",
        s3: String? = null,
        selfieS3: String? = "https://b.s3.amazonaws.com/r.jpg",
        selfieB64: String? = null,
    ) = ScoreRequest.validate(apiBase, checks, path, s3, selfieS3, selfieB64, null, null)

    private fun invalid(block: () -> Unit) {
        try { block(); fail("expected invalid_config") } catch (e: ScoreException) { assertEquals("invalid_config", e.code) }
    }

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
        try { block(); fail("expected server") } catch (e: ScoreException) { assertEquals("server", e.code) }
    }
}
