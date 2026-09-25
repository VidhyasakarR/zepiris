import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

/// Liveness challenge the rider must complete before the photo is taken.
enum LsnChallenge {
  none,
  blink,
  turn,

  /// blink or turn, picked per capture
  random;

  String get wire => name;
}

/// What the host app asks the capture flow to do.
class LsnCaptureConfig {
  LsnCaptureConfig({
    required String apiBase,
    this.checks = const ['face_match', 'dress_color', 'logo'],
    this.sourceSelfieS3,
    this.sourceSelfieB64,
    this.challenge = LsnChallenge.blink,
    this.submit = true,
    this.qualityCheck = true,
    this.headers = const {},
    this.httpClient,
    this.qualityTimeout = const Duration(seconds: 8),
    this.scoreTimeout = const Duration(seconds: 40),
  }) : apiBase = apiBase.trim().replaceAll(RegExp(r'/+$'), '') {
    final uri = Uri.tryParse(this.apiBase);
    if (uri == null ||
        !(uri.isScheme('https') || uri.isScheme('http')) ||
        uri.host.isEmpty) {
      throw ArgumentError.value(apiBase, 'apiBase', 'must be an http(s) URL');
    }
    final unknown = checks.where((c) => !kLsnChecks.contains(c)).toList();
    if (checks.isEmpty || unknown.isNotEmpty) {
      throw ArgumentError.value(
          checks, 'checks', 'must be a non-empty subset of $kLsnChecks');
    }
    if (submit &&
        checks.contains('face_match') &&
        (sourceSelfieS3 ?? '').isEmpty &&
        (sourceSelfieB64 ?? '').isEmpty) {
      throw ArgumentError('face_match needs sourceSelfieS3 or sourceSelfieB64');
    }
  }

  /// Server root, e.g. https://3-108-193-187.sslip.io (trailing slashes dropped).
  final String apiBase;

  /// Any of [kLsnChecks].
  final List<String> checks;

  /// Enrolled selfie for face_match: an S3/HTTP link, or a base64 JPEG.
  final String? sourceSelfieS3, sourceSelfieB64;

  final LsnChallenge challenge;

  /// true: score the photo (POST /v1/checkpoint/score) and return the scores.
  /// false: return the photo only; the host scores it itself.
  final bool submit;

  /// Run the server's photo quality check (blur, light, T-shirt in view) first,
  /// and let the rider retake on a warning. Advisory: never blocks a capture.
  final bool qualityCheck;

  /// Extra request headers (e.g. an auth token for your gateway).
  final Map<String, String> headers;

  /// HTTP client for every call the SDK makes: pass an inspector's client
  /// (e.g. Chucker's) to see them, or one with your own interceptors.
  final http.Client? httpClient;

  final Duration qualityTimeout, scoreTimeout;
}

const kLsnChecks = ['face_match', 'dress_color', 'logo'];

/// /v1/checkpoint/score, typed. Scores only: pass / fail is the host
/// backend's decision (e.g. the go-online API).
class LsnScores {
  const LsnScores(this.json);

  /// The response exactly as the server sent it (send this on, unmodified).
  final Map<String, dynamic> json;

  String get requestId => json['requestId'] as String? ?? '';
  DateTime? get scoredAt =>
      DateTime.tryParse(json['scoredAt'] as String? ?? '');
  List<String> get checksRequested =>
      ((json['checksRequested'] as List?) ?? const []).cast<String>();
  Map<String, dynamic> _s(String k) =>
      ((json['scores'] as Map?)?[k] as Map?)?.cast<String, dynamic>() ??
      const {};
  double? _d(Object? v) => (v as num?)?.toDouble();

  /// Cosine similarity to the enrolled selfie (null: no face found / not requested).
  double? get faceSimilarity => _d(_s('face_match')['similarity']);

  /// Probability the selfie is a live person (null when liveness is off).
  double? get liveness => _d(_s('face_match')['liveness']);
  bool? get faceDetected => _s('face_match')['faceDetected'] as bool?;
  double? get dressColor => _d(_s('dress_color')['score']);
  double? get logo => _d(_s('logo')['score']);

  /// What the logo check read on the shirt: {text, matched, onBlue, reason}.
  Map<String, dynamic>? get logoRead =>
      (_s('logo')['read'] as Map?)?.cast<String, dynamic>();

  /// SHA-256 (hex) of the exact photo that was scored.
  String? get imageSha256 => (json['image'] as Map?)?['sha256'] as String?;
}

/// What the flow hands back. `null` from LsnCapture.start = the rider backed out.
class LsnCaptureResult {
  const LsnCaptureResult({
    required this.photoJpeg,
    required this.challenge,
    required this.challengePassed,
    this.quality,
    this.scores,
    this.stats = const {},
  });

  /// The exact photo that was scored (SHA-256 = [LsnScores.imageSha256]).
  final Uint8List photoJpeg;

  /// The challenge that ran on the phone (random resolved to blink / turn).
  final String challenge;
  final bool challengePassed;

  /// /v1/quality/check response ({ok, warnings, face, shirt}), or null.
  final Map<String, dynamic>? quality;

  /// Null only when [LsnCaptureConfig.submit] is false.
  final LsnScores? scores;

  /// On-device timings: firstDetectionMs, detectMsAvg, captureMs, frames, flashed.
  final Map<String, dynamic> stats;

  String? get requestId => scores?.requestId;

  /// Everything except the photo, e.g. to log or to send to your backend.
  Map<String, dynamic> toJson() => {
        'challenge': challenge,
        'challengePassed': challengePassed,
        'quality': quality,
        'scores': scores?.json,
        'stats': stats,
        'photoBytes': photoJpeg.length,
      };
}

class LsnApiError implements Exception {
  LsnApiError(this.message, [this.status]);
  final String message;
  final int? status;
  @override
  String toString() => message;
}

/// The two backend calls the flow makes.
class CheckpointApi {
  CheckpointApi(this.config)
      : _http = config.httpClient ?? http.Client(),
        _ownsClient = config.httpClient == null;
  final LsnCaptureConfig config;
  final http.Client _http;
  final bool _ownsClient;

  Map<String, String> get _headers => {
        'Content-Type': 'application/json',
        'User-Agent': 'LSNCaptureSDK/1.1',
        ...config.headers,
      };

  Future<Map<String, dynamic>> _post(
      String path, Map<String, dynamic> body, Duration timeout) async {
    final http.Response r;
    try {
      r = await _http
          .post(Uri.parse('${config.apiBase}$path'),
              headers: _headers, body: jsonEncode(body))
          .timeout(timeout);
    } on TimeoutException {
      throw LsnApiError(
          'The server took too long. Check the connection and try again.');
    } catch (_) {
      throw LsnApiError(
          'Could not reach the server. Check the connection and try again.');
    }
    Object? d;
    try {
      d = jsonDecode(r.body);
    } catch (_) {}
    if (r.statusCode >= 400) {
      final det = d is Map ? d['detail'] : null;
      final why = switch (det) {
        String s => s,
        Map m => m['message'] ?? jsonEncode(m),
        List l when l.isNotEmpty =>
          (l.first is Map ? (l.first as Map)['msg'] : null) ??
              'invalid request',
        _ => r.reasonPhrase ?? 'error',
      };
      throw LsnApiError('Server error ${r.statusCode}: $why', r.statusCode);
    }
    if (d is! Map<String, dynamic>) {
      // e.g. a proxy's HTML page: never read that as scores
      throw LsnApiError(
          'The server sent an unexpected reply (${r.statusCode}).',
          r.statusCode);
    }
    return d;
  }

  void close() {
    if (_ownsClient) _http.close(); // never close a client the host passed in
  }

  /// Blur / light / T-shirt warnings (null if unreachable: advisory only).
  Future<Map<String, dynamic>?> quality(Uint8List jpeg) async {
    try {
      return await _post('/v1/quality/check', {'image_b64': base64Encode(jpeg)},
          config.qualityTimeout);
    } catch (_) {
      return null;
    }
  }

  /// Scores only (no decision) for the photo.
  Future<LsnScores> score(Uint8List jpeg) async {
    final d = await _post(
        '/v1/checkpoint/score',
        {
          'face_check_b64': base64Encode(jpeg),
          'checks': config.checks,
          if ((config.sourceSelfieS3 ?? '').isNotEmpty)
            'source_selfie_s3': config.sourceSelfieS3,
          if ((config.sourceSelfieB64 ?? '').isNotEmpty)
            'source_selfie_b64': config.sourceSelfieB64,
        },
        config.scoreTimeout);
    if (d['scores'] is! Map) throw LsnApiError('The server sent no scores.');
    return LsnScores(d);
  }
}
