import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'api.dart';

const _bg = Color(0xFF05080F);
const _ok = Color(0xFF0F8A5F);
const _bad = Color(0xFFC43B2F);
const _muted = Color(0xFF26324D);

/// Native side (Android): warm-up and the capture screen (CameraX + ML Kit).
class NativeCapture {
  static const _ch = MethodChannel('lsn_capture_sdk');

  static Future<Map<String, dynamic>> warmUp() async =>
      Map<String, dynamic>.from(await _ch.invokeMethod<Map>('warmUp') ?? {});

  /// The captured photo file and on-device stats, or null if the rider backed out.
  static Future<({String path, Map<String, dynamic> stats})?> capture(
    String challenge, {
    bool debug = false,
  }) async {
    final r = await _ch
        .invokeMethod<Map>('capture', {'challenge': challenge, 'debug': debug});
    if (r == null || r['path'] == null) return null;
    var stats = <String, dynamic>{};
    try {
      stats = jsonDecode(r['stats'] as String) as Map<String, dynamic>;
    } catch (_) {}
    return (path: r['path'] as String, stats: stats);
  }
}

enum _Stage { capturing, checking, review, scoring, error }

/// camera → photo quality check → scores → back to the host.
///
/// It stops on the photo only when the rider has a choice to make: a quality
/// warning (Retake / Use anyway) or a failed scoring call (Retake / Try again).
/// Otherwise the scores return to the host straight away.
class CaptureScreen extends StatefulWidget {
  const CaptureScreen({
    super.key,
    required this.config,
    this.accent = const Color(0xFF1F5FD6),
    this.debug = false,
  });
  final LsnCaptureConfig config;
  final Color accent;
  final bool debug;

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  late final _api = CheckpointApi(widget.config);
  _Stage _stage = _Stage.capturing;
  Uint8List? _photo;
  Map<String, dynamic> _stats = {};
  Map<String, dynamic>? _quality;
  String? _error;
  int _attempt = 0; // late replies for an older photo are dropped
  bool _done = false;

  List<Map> get _warnings =>
      (_quality?['warnings'] as List?)?.cast<Map>() ?? const [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _shoot());
  }

  @override
  void dispose() {
    _api.close();
    super.dispose();
  }

  Future<void> _shoot() async {
    final mine = ++_attempt;
    setState(() {
      _stage = _Stage.capturing;
      _photo = null;
      _quality = null;
      _error = null;
    });
    try {
      final shot = await NativeCapture.capture(widget.config.challenge.wire,
          debug: widget.debug);
      final file = shot == null ? null : File(shot.path);
      if (!mounted || mine != _attempt) {
        file
            ?.delete()
            .ignore(); // a newer attempt / closed screen: drop this photo
        return;
      }
      if (file == null) {
        return Navigator.of(context).pop(); // backed out of the camera
      }
      final Uint8List bytes;
      try {
        bytes = await file.readAsBytes();
      } finally {
        file.delete().ignore(); // the selfie never stays on disk
      }
      if (!mounted || mine != _attempt) return;
      setState(() {
        _photo = bytes;
        _stats = shot!.stats;
        _stage = widget.config.qualityCheck ? _Stage.checking : _Stage.review;
      });
      if (widget.config.qualityCheck) {
        final q = await _api.quality(bytes);
        if (!mounted || mine != _attempt) return;
        setState(() => _quality = q);
        if (_warnings.isNotEmpty) return setState(() => _stage = _Stage.review);
      }
      await _use(); // clear photo: no need to ask
    } on PlatformException catch (e) {
      if (!mounted) return;
      setState(() {
        _stage = _Stage.error;
        _error = switch (e.code) {
          'no_permission' =>
            'Camera permission is needed. Allow it in Settings.',
          'face_check_unavailable' =>
            'The face check could not start on this phone (it needs Google Play Services).',
          _ => 'Camera unavailable (${e.message ?? e.code}).',
        };
      });
    }
  }

  /// Score the photo (or hand it back unscored when submit is off).
  Future<void> _use() async {
    final photo = _photo;
    if (photo == null || _stage == _Stage.scoring) return;
    if (!widget.config.submit) return _finish(null);
    final mine = _attempt;
    setState(() {
      _stage = _Stage.scoring;
      _error = null;
    });
    try {
      final scores = await _api.score(photo);
      if (mounted && mine == _attempt) _finish(scores);
    } on LsnApiError catch (e) {
      if (mounted && mine == _attempt) {
        setState(() {
          _error = e.message;
          _stage = _Stage.review;
        });
      }
    }
  }

  void _finish(LsnScores? scores) {
    if (_done) return;
    _done = true;
    Navigator.of(context).pop(
      LsnCaptureResult(
        photoJpeg: _photo!,
        challenge:
            _stats['challenge'] as String? ?? widget.config.challenge.wire,
        challengePassed: _stats['challengePassed'] as bool? ?? false,
        quality: _quality,
        scores: scores,
        stats: _stats,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: _stage != _Stage.scoring, // never drop a scoring call in flight
      child: Scaffold(
        backgroundColor: _bg,
        body: Stack(
          fit: StackFit.expand,
          children: [
            if (_photo != null)
              // The true scene, not mirrored, exactly as sent to the server
              // (the live preview is unmirrored too, so the two match).
              Center(
                  child: Image.memory(_photo!,
                      fit: BoxFit.contain, gaplessPlayback: true))
            else if (_stage == _Stage.capturing)
              const Center(child: CircularProgressIndicator()),
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: SafeArea(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(6, 6, 16, 0),
                  child: Row(
                    children: [
                      IconButton.filledTonal(
                        onPressed: _stage == _Stage.scoring
                            ? null
                            : () => Navigator.of(context).pop(),
                        icon: const Icon(Icons.arrow_back),
                        tooltip: 'Back',
                      ),
                      const SizedBox(width: 10),
                      const Text(
                        'Checkpoint selfie',
                        style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w700,
                            fontSize: 16),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: SafeArea(
                child: Container(
                  padding: const EdgeInsets.fromLTRB(16, 14, 16, 12),
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [Color(0x0005080F), Color(0xE605080F)],
                    ),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: _dock(),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _dock() {
    Widget banner(String t, {Color? color}) => Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
          margin: const EdgeInsets.only(bottom: 10),
          decoration: BoxDecoration(
              color: color ?? const Color(0xCC0F1728),
              borderRadius: BorderRadius.circular(22)),
          child: Text(
            t,
            textAlign: TextAlign.center,
            style: const TextStyle(
                color: Colors.white, fontSize: 15, fontWeight: FontWeight.w600),
          ),
        );
    ButtonStyle big(Color c) => FilledButton.styleFrom(
          backgroundColor: c,
          foregroundColor: Colors.white,
          disabledBackgroundColor: _muted,
          disabledForegroundColor: Colors.white70,
          minimumSize: const Size.fromHeight(54),
          textStyle: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
        );

    switch (_stage) {
      case _Stage.capturing:
        return [banner('Opening the camera…')];
      case _Stage.checking:
        return [banner('Checking the photo…')];
      case _Stage.scoring:
        return [banner('Getting your scores…')];
      case _Stage.error:
        return [
          banner(_error ?? 'Something went wrong', color: _bad),
          FilledButton(
              style: big(widget.accent),
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Back')),
        ];
      case _Stage.review:
        final warnings = _warnings;
        return [
          if (warnings.isNotEmpty)
            banner(
              'Retake recommended:\n${warnings.map((w) => '• ${w['message']}').join('\n')}',
              color: const Color(0xE63A1512),
            ),
          if (_error != null) banner(_error!, color: _bad),
          if (warnings.isEmpty && _error == null)
            banner('Photo clear', color: _ok),
          Row(
            children: [
              Expanded(
                child: FilledButton(
                  style: big(warnings.isNotEmpty ? widget.accent : _muted),
                  onPressed: _shoot,
                  child: const Text('Retake'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: FilledButton(
                  style: big(warnings.isNotEmpty ? _muted : widget.accent),
                  onPressed: _use,
                  child: Text(_error != null
                      ? 'Try again'
                      : warnings.isNotEmpty
                          ? 'Use anyway'
                          : 'Use photo'),
                ),
              ),
            ],
          ),
        ];
    }
  }
}
