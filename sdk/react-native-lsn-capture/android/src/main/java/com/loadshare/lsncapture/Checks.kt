package com.loadshare.lsncapture

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Live checks and the liveness challenge on plain numbers (no camera code), so
 * they are unit-tested (see src/test) and identical on every device.
 *
 * Coordinates are normalised to the upright analysis frame (0..1, origin top
 * left). That frame is 4:3, the same field of view as the preview and the photo,
 * so what the checks see is what the rider sees and what the server gets.
 * Angles are ML Kit's head Euler angles in degrees.
 */
data class FaceSample(
    val left: Float,
    val top: Float,
    val width: Float,
    val height: Float,
    val yaw: Float = 0f,
    val pitch: Float = 0f,
    val roll: Float = 0f,
    val leftEyeOpen: Float? = null,
    val rightEyeOpen: Float? = null,
) {
    val cx get() = left + width / 2
    val cy get() = top + height / 2

    /** The more-open eye: both must close for a blink. */
    val eyesOpenMax: Float? get() = if (leftEyeOpen == null || rightEyeOpen == null) null else max(leftEyeOpen, rightEyeOpen)

    /** The more-closed eye: "open your eyes" if either is shut. */
    val eyesOpenMin: Float? get() = if (leftEyeOpen == null || rightEyeOpen == null) null else min(leftEyeOpen, rightEyeOpen)
}

/**
 * Thresholds in one place. The T-shirt rule is the server's capture quality
 * check (capture_quality.py): the region runs from just under the chin to three
 * face heights below the top of the head, and 35% of it must be in frame, so a
 * close selfie that shows the chest passes.
 */
object Limits {
    const val MIN_FACE_WIDTH = 0.16f      // smaller: too far to verify
    const val MAX_FACE_WIDTH = 0.62f      // bigger: the T-shirt cannot be in frame
    const val MAX_CENTRE_OFFSET = 0.18f
    const val SHIRT_FACE_HEIGHTS = 3.0f
    const val MIN_SHIRT_VISIBLE = 0.35f
    const val MAX_YAW = 12f
    const val MAX_PITCH = 14f
    const val MAX_ROLL = 15f
    const val EYES_OPEN = 0.35f          // lower with glasses / dim light; blinks are judged relatively
    // Darkness only BLOCKS when the picture is nearly black. Anything the face
    // detector can find is capturable: dim light is a warning ("Low light —
    // tap 💡"), and the server's quality check has the final word.
    const val TOO_DARK = 15f              // face luma (0-255): below this nothing can be made out
    const val SHIRT_TOO_DARK = 35f        // T-shirt luma (75th pct): warning only
    const val LOW_LIGHT = 90f
    const val TOO_BRIGHT = 225f
    const val STILL = 0.03f               // face centre travel over ~0.4 s, fraction of the frame
}

/** Share (0..1) of the chin-to-stomach T-shirt region inside the frame. */
fun shirtVisible(f: FaceSample): Float {
    val top = f.top + f.height * 1.3f
    val bottom = f.top + f.height * Limits.SHIRT_FACE_HEIGHTS
    if (bottom <= top) return 1f
    return ((min(1f, bottom) - min(1f, top)) / (bottom - top)).coerceIn(0f, 1f)
}

/** What is wrong right now, most important first; null when all is well. */
fun liveProblem(
    faces: List<FaceSample>,
    luma: Float? = null,
    shirtLuma: Float? = null,
    movement: Float = 0f,
    turning: Boolean = false,
    blinking: Boolean = false,
): String? {
    if (luma != null && luma < Limits.TOO_DARK) return "Too dark — move to a brighter place"
    if (faces.isEmpty()) return "Look at the camera"
    if (faces.size > 1) return "Only one person in the frame"
    if (luma != null && luma > Limits.TOO_BRIGHT) return "Too bright — move out of direct light"
    val f = faces[0]
    if (f.width < Limits.MIN_FACE_WIDTH) return "Come a little closer"
    if (f.width > Limits.MAX_FACE_WIDTH) return "Move the phone a little further away"
    if (abs(f.cx - 0.5f) > Limits.MAX_CENTRE_OFFSET) return "Move your face to the centre"
    if (shirtVisible(f) < Limits.MIN_SHIRT_VISIBLE) return "Hold the phone further away — show your T-shirt"
    if (!turning && abs(f.yaw) > Limits.MAX_YAW) return "Look straight at the camera"
    if (abs(f.pitch) > Limits.MAX_PITCH) return "Keep your head level"
    if (abs(f.roll) > Limits.MAX_ROLL) return "Keep your head straight"
    if (!blinking && (f.eyesOpenMin ?: 1f) < Limits.EYES_OPEN) return "Open your eyes"
    if (!turning && !blinking && movement > Limits.STILL) return "Hold still"
    return null
}

enum class ChallengeType { NONE, BLINK, TURN;
    companion object {
        fun from(name: String?, coin: () -> Boolean = { Math.random() < 0.5 }) = when (name) {
            "blink" -> BLINK
            "turn" -> TURN
            "random" -> if (coin()) BLINK else TURN
            else -> NONE
        }
    }
}

/**
 * The liveness challenge: blink, or turn the head to one side and back.
 *
 * Blink is measured against the rider's own open-eye level (glasses and dim
 * light keep ML Kit's "open" score lower): closed = the more-open eye drops
 * below [closedAt]; open again = the more-closed eye rises above [openAt]. A
 * still photo never changes; a wink closes only one eye.
 * Turn: |yaw| beyond TURNED, then back within FRONTAL.
 */
class Challenge(var type: ChallengeType) {
    var phase = 0
        private set
    var done = type == ChallengeType.NONE
        private set
    private var base: Float? = null
    private var closedFrames = 0

    fun reset(to: ChallengeType? = null) {
        if (to != null) type = to
        phase = 0
        done = type == ChallengeType.NONE
        base = null
    }

    fun closedAt() = ((base ?: 0.9f) * 0.45f).coerceIn(0.15f, 0.35f)
    fun openAt() = max(0.5f, (base ?: 0.9f) * 0.75f)

    val instruction: String
        get() = when (type) {
            ChallengeType.BLINK -> "Blink 3 times, quickly" // one blink passes; three give several chances
            ChallengeType.TURN -> if (phase == 0) "Turn your head to one side" else "Now look back at the camera"
            ChallengeType.NONE -> ""
        }

    /** Feed one frame's single face; true once done. */
    fun step(f: FaceSample): Boolean {
        if (done) return true
        when (type) {
            ChallengeType.BLINK -> {
                val both = f.eyesOpenMax ?: return false
                val either = f.eyesOpenMin ?: return false
                if (phase == 0) {
                    if (base == null) base = both
                    if (both < closedAt()) { phase = 1; closedFrames = 0 }
                    else base = 0.85f * base!! + 0.15f * both // follow the open level
                } else if (either > openAt()) {
                    done = true
                } else if (++closedFrames > MAX_CLOSED_FRAMES) {
                    phase = 0 // eyes kept shut (or a bad frame run): that was no blink
                }
            }
            ChallengeType.TURN -> {
                if (phase == 0 && abs(f.yaw) > TURNED) phase = 1
                else if (phase == 1 && abs(f.yaw) < FRONTAL) done = true
            }
            ChallengeType.NONE -> done = true
        }
        return done
    }

    companion object {
        const val TURNED = 20f
        const val FRONTAL = 8f
        /** ~1.5 s of analysis frames: a blink reopens well within that. */
        const val MAX_CLOSED_FRAMES = 30
    }
}
