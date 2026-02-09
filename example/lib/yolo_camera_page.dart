import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

import 'yolo_detector.dart';

class YoloCameraPage extends StatefulWidget {
  final String modelPath;
  final List<OrtProvider>? providers;
  final Map<OrtProvider, Map<String, String>> providerOptions;

  const YoloCameraPage({
    super.key,
    required this.modelPath,
    this.providers,
    this.providerOptions = const {},
  });

  @override
  State<YoloCameraPage> createState() => _YoloCameraPageState();
}

class _YoloCameraPageState extends State<YoloCameraPage>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  YoloDetector? _detector;
  List<Detection> _detections = [];
  bool _isProcessing = false;
  bool _initializing = true;
  String? _error;
  double _fps = 0;
  double _inferenceMs = 0;
  double _confThreshold = 0.25;
  int _frameCount = 0;
  DateTime _lastFpsTime = DateTime.now();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      // Load YOLO model
      _detector = YoloDetector.create(
        widget.modelPath,
        providers: widget.providers,
        providerOptions: widget.providerOptions,
      );

      // Get cameras
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _error = 'No cameras available';
          _initializing = false;
        });
        return;
      }

      // Prefer back camera
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888,
      );

      await _cameraController!.initialize();

      if (!mounted) return;

      await _cameraController!.startImageStream(_onCameraFrame);

      setState(() => _initializing = false);
    } catch (e) {
      setState(() {
        _error = e.toString();
        _initializing = false;
      });
    }
  }

  void _onCameraFrame(CameraImage cameraImage) {
    if (_isProcessing || _detector == null) return;
    _isProcessing = true;

    try {
      final sw = Stopwatch()..start();
      final detections = _detector!.detectFromCameraImage(
        cameraImage,
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
      debugPrint('Detection error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initialize();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('YOLO Real-Time Detection'),
        backgroundColor: Colors.black87,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.tune),
            onPressed: _showSettingsSheet,
          ),
        ],
      ),
      body: _initializing
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
          : _buildCameraView(),
    );
  }

  Widget _buildCameraView() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(
        child: Text(
          'Camera not available',
          style: TextStyle(color: Colors.white70),
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera preview
        Center(
          child: AspectRatio(
            aspectRatio: _cameraController!.value.aspectRatio,
            child: CameraPreview(_cameraController!),
          ),
        ),

        // Detection overlay
        Center(
          child: AspectRatio(
            aspectRatio: _cameraController!.value.aspectRatio,
            child: LayoutBuilder(
              builder: (context, constraints) {
                return CustomPaint(
                  size: Size(constraints.maxWidth, constraints.maxHeight),
                  painter: DetectionPainter(
                    detections: _detections,
                    previewWidth: constraints.maxWidth,
                    previewHeight: constraints.maxHeight,
                    // Camera image is rotated 90° on Android
                    rotated: Platform.isAndroid,
                  ),
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

        // Detection list at bottom
        if (_detections.isNotEmpty)
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: Container(
              constraints: const BoxConstraints(maxHeight: 120),
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
// Detection overlay painter
// ---------------------------------------------------------------------------

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final double previewWidth;
  final double previewHeight;
  final bool rotated;

  DetectionPainter({
    required this.detections,
    required this.previewWidth,
    required this.previewHeight,
    this.rotated = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    for (final det in detections) {
      double left, top, right, bottom;

      if (rotated) {
        // On Android the camera image is landscape (rotated 90°),
        // so we swap x/y to map onto the portrait preview.
        left = det.y1 * size.width;
        top = (1.0 - det.x2) * size.height;
        right = det.y2 * size.width;
        bottom = (1.0 - det.x1) * size.height;
      } else {
        left = det.x1 * size.width;
        top = det.y1 * size.height;
        right = det.x2 * size.width;
        bottom = det.y2 * size.height;
      }

      final rect = Rect.fromLTRB(left, top, right, bottom);

      final color = _colorForClass(det.classId);

      // Box
      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(4)),
        boxPaint,
      );

      // Label background
      final labelText =
          '${det.label} ${(det.confidence * 100).toStringAsFixed(0)}%';
      final textSpan = TextSpan(
        text: labelText,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final bgRect = Rect.fromLTWH(
        left,
        top - textPainter.height - 4,
        textPainter.width + 8,
        textPainter.height + 4,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
        Paint()..color = color,
      );
      textPainter.paint(canvas, Offset(left + 4, top - textPainter.height - 2));
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
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
