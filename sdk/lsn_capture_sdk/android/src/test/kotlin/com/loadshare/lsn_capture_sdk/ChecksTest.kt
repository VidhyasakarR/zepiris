package com.loadshare.lsn_capture_sdk

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ChecksTest {
    // A well-framed rider in the 4:3 portrait frame: face 30% wide, upper centre.
    private fun good(
        yaw: Float = 0f, pitch: Float = 0f, l: Float = 0.95f, r: Float = 0.95f,
        top: Float = 0.14f, width: Float = 0.30f, left: Float = 0.35f,
    ) = FaceSample(left, top, width, width * 0.95f * 3f / 4f, yaw, pitch, 0f, l, r) // box ~square in pixels; frame is 3:4

    @Test fun wellFramedPasses() = assertNull(liveProblem(listOf(good()), luma = 120f))
    @Test fun closeSelfieShowingChestPasses() = assertNull(liveProblem(listOf(good(width = 0.5f, left = 0.25f, top = 0.12f)), luma = 120f))
    @Test fun noFace() = assertEquals("Look at the camera", liveProblem(emptyList(), luma = 120f))
    @Test fun twoFaces() = assertEquals("Only one person in the frame", liveProblem(listOf(good(), good(left = 0.05f)), luma = 120f))
    @Test fun tooDarkFirst() = assertTrue(liveProblem(listOf(good()), luma = 30f)!!.startsWith("Too dark"))
    @Test fun tooFar() = assertEquals("Come a little closer", liveProblem(listOf(good(width = 0.1f)), luma = 120f))
    @Test fun tooClose() = assertTrue(liveProblem(listOf(good(width = 0.7f, left = 0.15f)), luma = 120f)!!.startsWith("Move the phone"))
    @Test fun offCentre() = assertEquals("Move your face to the centre", liveProblem(listOf(good(left = 0.02f)), luma = 120f))
    @Test fun shirtCutOff() = assertTrue(liveProblem(listOf(good(top = 0.7f)), luma = 120f)!!.contains("T-shirt"))
    @Test fun headTurned() = assertEquals("Look straight at the camera", liveProblem(listOf(good(yaw = 25f)), luma = 120f))
    @Test fun turningOkDuringTurn() = assertNull(liveProblem(listOf(good(yaw = 25f)), luma = 120f, turning = true))
    @Test fun eyesShut() = assertEquals("Open your eyes", liveProblem(listOf(good(l = 0.1f, r = 0.1f)), luma = 120f))
    @Test fun eyesShutOkMidBlink() = assertNull(liveProblem(listOf(good(l = 0.1f, r = 0.1f)), luma = 120f, blinking = true))
    @Test fun moving() = assertEquals("Hold still", liveProblem(listOf(good()), luma = 120f, movement = 0.08f))

    private fun run(t: ChallengeType, frames: List<FaceSample>): Boolean {
        val c = Challenge(t)
        frames.forEach { c.step(it) }
        return c.done
    }

    @Test fun blinkPasses() = assertTrue(run(ChallengeType.BLINK, listOf(good(), good(l = 0.05f, r = 0.08f), good())))
    @Test fun blinkBehindGlasses() = // open level ~0.6, blink bottoms at 0.25
        assertTrue(run(ChallengeType.BLINK, listOf(good(l = 0.62f, r = 0.6f), good(l = 0.6f, r = 0.58f), good(l = 0.24f, r = 0.25f), good(l = 0.6f, r = 0.61f))))
    @Test fun stillPhotoNeverBlinks() = assertFalse(run(ChallengeType.BLINK, List(200) { good() }))
    @Test fun jitterIsNotABlink() = assertFalse(run(ChallengeType.BLINK, List(200) { i -> good(l = 0.8f + (i % 5) * 0.04f, r = 0.85f) }))
    @Test fun winkIsNotABlink() = assertFalse(run(ChallengeType.BLINK, listOf(good(), good(l = 0.05f, r = 0.95f), good())))
    @Test fun eyesThatStayShut() = assertFalse(run(ChallengeType.BLINK, listOf(good(), good(l = 0.05f, r = 0.05f), good(l = 0.2f, r = 0.2f))))
    @Test fun turnPasses() = assertTrue(run(ChallengeType.TURN, listOf(good(), good(yaw = 28f), good(yaw = 3f))))
    @Test fun wobbleIsNotATurn() = assertFalse(run(ChallengeType.TURN, listOf(good(), good(yaw = 12f), good())))
    @Test fun noneIsDone() = assertTrue(Challenge(ChallengeType.NONE).done)
    @Test fun resetStartsOver() {
        val c = Challenge(ChallengeType.BLINK)
        listOf(good(), good(l = 0f, r = 0f), good()).forEach { c.step(it) }
        assertTrue(c.done)
        c.reset()
        assertFalse(c.done)
    }
    @Test fun names() {
        assertEquals(ChallengeType.BLINK, ChallengeType.from("blink"))
        assertEquals(ChallengeType.TURN, ChallengeType.from("turn"))
        assertEquals(ChallengeType.NONE, ChallengeType.from("none"))
        assertEquals(ChallengeType.TURN, ChallengeType.from("random") { false })
    }
}
