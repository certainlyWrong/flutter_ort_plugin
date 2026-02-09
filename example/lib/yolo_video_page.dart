import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';
import 'package:image/image.dart' as img;
import 'package:video_player/video_player.dart';

import 'yolo_detector.dart';

class YoloVideoPage extends StatefulWidget {
  final String modelPath;
  final List<OrtProvider>? providers;
  final Map<OrtProvider, Map<String, String>> providerOptions;

  const YoloVideoPage({
    super.key,
    required this.modelPath,
    this.providers,
    this.providerOptions = const {},
  });

  @override
  State<YoloVideoPage> createState() => _YoloVideoPageState();
}

class _YoloVideoPageState extends State<YoloVideoPage> {
  final GlobalKey _videoBoundaryKey = GlobalKey();
  VideoPlayerController? _controller;
  YoloDetector? _detector;
  Timer? _inferenceTimer;
  bool _loading = true;
  String? _error;
  bool _isPlaying = false;
  bool _isProcessing = false;
  List<Detection> _detections = [];
  double _confThreshold = 0.25;
  double _inferenceMs = 0;
  int _frameCount = 0;
  double _fps = 0;
  DateTime _lastFpsTime = DateTime.now();

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      _detector = YoloDetector.create(
        widget.modelPath,
        providers: widget.providers,
        providerOptions: widget.providerOptions,
      );

      // Copy video from assets to temp
      final videoBytes = await rootBundle.load('assets/street.mp4');
      final tempDir = Directory.systemTemp;
      final videoFile = File('${tempDir.path}/yolo_street.mp4');
      await videoFile.writeAsBytes(videoBytes.buffer.asUint8List());

      _controller = VideoPlayerController.file(videoFile);
      await _controller!.initialize();
      _controller!.setLooping(true);

      setState(() => _loading = false);
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _togglePlayPause() {
    if (_controller == null) return;

    if (_isPlaying) {
      _controller!.pause();
      _inferenceTimer?.cancel();
      _inferenceTimer = null;
    } else {
      _controller!.play();
      // Capture a frame every ~200ms for inference
      _inferenceTimer = Timer.periodic(
        const Duration(milliseconds: 200),
        (_) => _captureAndDetect(),
      );
    }

    setState(() => _isPlaying = !_isPlaying);
  }

  Future<void> _captureAndDetect() async {
    if (_isProcessing || _detector == null || !mounted) return;
    _isProcessing = true;

    try {
      final boundary =
          _videoBoundaryKey.currentContext?.findRenderObject()
              as RenderRepaintBoundary?;
      if (boundary == null || !boundary.hasSize) {
        _isProcessing = false;
        return;
      }

      // Capture the rendered video widget as an image
      final uiImage = await boundary.toImage(pixelRatio: 1.0);
      final byteData = await uiImage.toByteData(
        format: ui.ImageByteFormat.rawRgba,
      );
      uiImage.dispose();

      if (byteData == null) {
        _isProcessing = false;
        return;
      }

      final width = boundary.size.width.toInt();
      final height = boundary.size.height.toInt();
      final pixels = byteData.buffer.asUint8List();

      // Convert RGBA to img.Image
      final frame = img.Image(width: width, height: height);
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final i = (y * width + x) * 4;
          frame.setPixelRgba(
            x,
            y,
            pixels[i],
            pixels[i + 1],
            pixels[i + 2],
            pixels[i + 3],
          );
        }
      }

      final sw = Stopwatch()..start();
      final detections = _detector!.detectFromImage(
        frame,
        confThreshold: _confThreshold,
      );
      sw.stop();

      _frameCount++;
      final now = DateTime.now();
      final elapsed = now.difference(_lastFpsTime).inMilliseconds;
      if (elapsed >= 1000) {
        _fps = _frameCount * 1000.0 / elapsed;
        _frameCount = 0;
        _lastFpsTime = now;
      }

      if (mounted) {
        setState(() {
          _detections = detections;
          _inferenceMs = sw.elapsedMicroseconds / 1000.0;
        });
      }
    } catch (e) {
      debugPrint('Frame capture error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  @override
  void dispose() {
    _inferenceTimer?.cancel();
    _controller?.dispose();
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('YOLO Video Detection'),
        backgroundColor: Colors.black87,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.tune),
            onPressed: _showSettingsSheet,
          ),
        ],
      ),
      body: _loading
          ? const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(color: Colors.white),
                  SizedBox(height: 16),
                  Text(
                    'Loading YOLO model...',
                    style: TextStyle(color: Colors.white70),
                  ),
                ],
              ),
            )
          : _error != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: SelectableText(
                  _error!,
                  style: const TextStyle(color: Colors.redAccent),
                  textAlign: TextAlign.center,
                ),
              ),
            )
          : _buildVideoView(),
    );
  }

  Widget _buildVideoView() {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Center(
        child: Text(
          'Video not available',
          style: TextStyle(color: Colors.white70),
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // Video player wrapped in RepaintBoundary for frame capture
        Center(
          child: AspectRatio(
            aspectRatio: _controller!.value.aspectRatio,
            child: RepaintBoundary(
              key: _videoBoundaryKey,
              child: VideoPlayer(_controller!),
            ),
          ),
        ),

        // Detection overlay
        Center(
          child: AspectRatio(
            aspectRatio: _controller!.value.aspectRatio,
            child: LayoutBuilder(
              builder: (context, constraints) {
                return CustomPaint(
                  size: Size(constraints.maxWidth, constraints.maxHeight),
                  painter: _VideoDetectionPainter(detections: _detections),
                );
              },
            ),
          ),
        ),

        // Stats overlay
        Positioned(
          left: 12,
          top: 12,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${_inferenceMs.toStringAsFixed(1)} ms',
                  style: const TextStyle(
                    color: Colors.greenAccent,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  '${_fps.toStringAsFixed(1)} FPS',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
                Text(
                  '${_detections.length} objects',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ),
          ),
        ),

        // Play/pause button
        Positioned(
          bottom: 20,
          right: 20,
          child: FloatingActionButton(
            onPressed: _togglePlayPause,
            backgroundColor: Colors.black54,
            child: Icon(
              _isPlaying ? Icons.pause : Icons.play_arrow,
              color: Colors.white,
            ),
          ),
        ),

        // Detection list at bottom
        if (_detections.isNotEmpty)
          Positioned(
            left: 0,
            right: 80,
            bottom: 0,
            child: Container(
              constraints: const BoxConstraints(maxHeight: 100),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.transparent, Colors.black87],
                ),
              ),
              child: ListView.builder(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                scrollDirection: Axis.horizontal,
                itemCount: _detections.length,
                itemBuilder: (context, index) {
                  final det = _detections[index];
                  return Container(
                    margin: const EdgeInsets.only(right: 8),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: _colorForClass(det.classId).withValues(alpha: 0.8),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          det.label,
                          style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          '${(det.confidence * 100).toStringAsFixed(0)}%',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 11,
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
          ),
      ],
    );
  }

  void _showSettingsSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.grey[900],
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setSheetState) {
            return Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Detection Settings',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 20),
                  Text(
                    'Confidence Threshold: ${(_confThreshold * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(color: Colors.white70),
                  ),
                  Slider(
                    value: _confThreshold,
                    min: 0.1,
                    max: 0.9,
                    divisions: 16,
                    activeColor: Colors.greenAccent,
                    onChanged: (v) {
                      setSheetState(() => _confThreshold = v);
                      setState(() => _confThreshold = v);
                    },
                  ),
                  const SizedBox(height: 12),
                ],
              ),
            );
          },
        );
      },
    );
  }

  Color _colorForClass(int classId) {
    const palette = [
      Color(0xFFE6194B),
      Color(0xFF3CB44B),
      Color(0xFFFFE119),
      Color(0xFF4363D8),
      Color(0xFFF58231),
      Color(0xFF911EB4),
      Color(0xFF42D4F4),
      Color(0xFFF032E6),
      Color(0xFFBFEF45),
      Color(0xFFFABED4),
      Color(0xFF469990),
      Color(0xFFDCBEFF),
      Color(0xFF9A6324),
      Color(0xFFFFFAC8),
      Color(0xFF800000),
      Color(0xFFAAFFC3),
      Color(0xFF808000),
      Color(0xFFFFD8B1),
      Color(0xFF000075),
      Color(0xFFA9A9A9),
    ];
    return palette[classId % palette.length];
  }
}

// ---------------------------------------------------------------------------
// Detection overlay painter for video
// ---------------------------------------------------------------------------

class _VideoDetectionPainter extends CustomPainter {
  final List<Detection> detections;

  _VideoDetectionPainter({required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    for (final det in detections) {
      final rect = Rect.fromLTRB(
        det.x1 * size.width,
        det.y1 * size.height,
        det.x2 * size.width,
        det.y2 * size.height,
      );

      final color = _colorForClass(det.classId);

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(4)),
        boxPaint,
      );

      final labelText =
          '${det.label} ${(det.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final bgRect = Rect.fromLTWH(
        rect.left,
        rect.top - textPainter.height - 4,
        textPainter.width + 8,
        textPainter.height + 4,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
        Paint()..color = color,
      );
      textPainter.paint(
        canvas,
        Offset(rect.left + 4, rect.top - textPainter.height - 2),
      );
    }
  }

  Color _colorForClass(int classId) {
    const palette = [
      Color(0xFFE6194B),
      Color(0xFF3CB44B),
      Color(0xFFFFE119),
      Color(0xFF4363D8),
      Color(0xFFF58231),
      Color(0xFF911EB4),
      Color(0xFF42D4F4),
      Color(0xFFF032E6),
      Color(0xFFBFEF45),
      Color(0xFFFABED4),
      Color(0xFF469990),
      Color(0xFFDCBEFF),
      Color(0xFF9A6324),
      Color(0xFFFFFAC8),
      Color(0xFF800000),
      Color(0xFFAAFFC3),
      Color(0xFF808000),
      Color(0xFFFFD8B1),
      Color(0xFF000075),
      Color(0xFFA9A9A9),
    ];
    return palette[classId % palette.length];
  }

  @override
  bool shouldRepaint(covariant _VideoDetectionPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
