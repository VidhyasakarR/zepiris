/// LSN Capture SDK: the Loadshare checkpoint selfie, native.
///
/// ```dart
/// // 1. Early (e.g. when the config / pre-capture screen shows):
/// LsnCapture.warmUp();
///
/// // 2. When the rider taps "Take selfie" (the app must hold CAMERA permission):
/// final r = await LsnCapture.start(context, LsnCaptureConfig(
///   apiBase: 'https://3-108-193-187.sslip.io',
///   checks: ['face_match', 'dress_color', 'logo'],
///   sourceSelfieS3: 'https://bucket.s3.amazonaws.com/rider.jpg',
///   challenge: LsnChallenge.blink,
/// ));
/// if (r != null) {
///   r.scores?.faceSimilarity;   // scores only — your backend decides
///   r.photoJpeg;                // the exact photo scored
/// }
/// ```
///
/// On device (Android, Kotlin): CameraX (preview + 640×480 analysis + 4:3
/// photo) and ML Kit face detection from Play Services for the live checks and
/// the blink / head-turn liveness challenge. The photo goes to the server's
/// /v1/quality/check and /v1/checkpoint/score. The SDK never decides pass /
/// fail; it returns the scores.
library;

import 'package:flutter/material.dart';

import 'src/api.dart';
import 'src/capture_screen.dart';

export 'src/api.dart'
    show
        LsnApiError,
        LsnCaptureConfig,
        LsnCaptureResult,
        LsnChallenge,
        LsnScores,
        kLsnChecks;

class LsnCapture {
  LsnCapture._();

  /// Get the face model ready (Play Services installs it if missing) and
  /// initialise CameraX, so [start] detects from the first camera frame.
  /// Returns {modelReady, ms, reason}. Safe to call more than once.
  static Future<Map<String, dynamic>> warmUp() => NativeCapture.warmUp();

  /// Run the capture flow full screen. Returns the result, or null if the
  /// rider backed out. The host app must already hold the CAMERA permission.
  /// [debug]: log per-frame face numbers to logcat (tag LsnCapture). Keep it
  /// off in release builds.
  static Future<LsnCaptureResult?> start(
    BuildContext context,
    LsnCaptureConfig config, {
    Color? accent,
    bool debug = false,
  }) {
    return Navigator.of(context).push<LsnCaptureResult>(
      MaterialPageRoute(
        builder: (_) => CaptureScreen(
            config: config,
            accent: accent ?? const Color(0xFF1F5FD6),
            debug: debug),
      ),
    );
  }
}
