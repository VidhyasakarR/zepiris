import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:lsn_capture_sdk/lsn_capture_sdk.dart';
import 'package:lsn_capture_sdk/src/api.dart' show CheckpointApi;

const _scores = {
  'requestId': 'r-1',
  'scoredAt': '2026-09-25T11:30:02Z',
  'checksRequested': ['face_match', 'dress_color', 'logo'],
  'scores': {
    'face_match': {'similarity': 0.82, 'liveness': 0.91, 'faceDetected': true},
    'dress_color': {'score': 0.96},
    'logo': {
      'score': 1.0,
      'read': {'text': 'LOADSHARE', 'matched': 'LOADSHARE', 'onBlue': 0.96, 'reason': null},
    },
  },
  'image': {'sha256': 'abc', 'bytes': 3},
};

void main() {
  test('config validates base, checks and source', () {
    expect(() => LsnCaptureConfig(apiBase: 'not a url'), throwsArgumentError);
    expect(() => LsnCaptureConfig(apiBase: 'https://x.dev', checks: ['shoes'], sourceSelfieS3: 's'), throwsArgumentError);
    expect(() => LsnCaptureConfig(apiBase: 'https://x.dev'), throwsArgumentError); // face_match, no source
    final c = LsnCaptureConfig(apiBase: ' https://x.dev/// ', checks: ['logo']);
    expect(c.apiBase, 'https://x.dev');
  });

  test('scores call: endpoint, body, headers, typed result', () async {
    late http.Request sent;
    final client = MockClient((req) async {
      sent = req;
      return http.Response(jsonEncode(_scores), 200, headers: {'content-type': 'application/json'});
    });
    final api = CheckpointApi(LsnCaptureConfig(
      apiBase: 'https://x.dev', sourceSelfieS3: 'https://s3/r.jpg', httpClient: client, headers: {'Authorization': 'Bearer t'},
    ));
    final s = await api.score(Uint8List.fromList([1, 2, 3]));
    expect(sent.url.toString(), 'https://x.dev/v1/checkpoint/score');
    expect(sent.headers['Authorization'], 'Bearer t');
    final body = jsonDecode(sent.body) as Map;
    expect(body['face_check_b64'], base64Encode([1, 2, 3]));
    expect(body['source_selfie_s3'], 'https://s3/r.jpg');
    expect(body['checks'], ['face_match', 'dress_color', 'logo']);
    expect(s.requestId, 'r-1');
    expect(s.faceSimilarity, 0.82);
    expect(s.liveness, 0.91);
    expect(s.dressColor, 0.96);
    expect(s.logo, 1.0);
    expect(s.logoRead?['text'], 'LOADSHARE');
    expect(s.imageSha256, 'abc');
    api.close(); // must not close the host's client
    expect(await client.get(Uri.parse('https://x.dev')), isA<http.Response>());
  });

  test('server errors become LsnApiError with the reason', () async {
    final api = CheckpointApi(LsnCaptureConfig(
      apiBase: 'https://x.dev', checks: ['logo'],
      httpClient: MockClient((_) async => http.Response(jsonEncode({'detail': {'message': 'dresscode_model_unavailable'}}), 503)),
    ));
    expect(() => api.score(Uint8List(1)), throwsA(isA<LsnApiError>().having((e) => e.status, 'status', 503)));
  });

  test('a proxy HTML page is never read as scores', () async {
    final api = CheckpointApi(LsnCaptureConfig(
      apiBase: 'https://x.dev', checks: ['logo'], httpClient: MockClient((_) async => http.Response('<html>', 200)),
    ));
    expect(() => api.score(Uint8List(1)), throwsA(isA<LsnApiError>()));
  });

  test('quality check never throws (advisory)', () async {
    final api = CheckpointApi(LsnCaptureConfig(
      apiBase: 'https://x.dev', checks: ['logo'], httpClient: MockClient((_) async => throw Exception('offline')),
    ));
    expect(await api.quality(Uint8List(1)), isNull);
  });
}
