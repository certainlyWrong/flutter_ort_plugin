import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';
import 'package:image/image.dart' as img;

import 'yolo_detector.dart';

class YoloImagePage extends StatefulWidget {
  final String modelPath;
  final List<OrtProvider>? providers;
  final Map<OrtProvider, Map<String, String>> providerOptions;
  final int inputSize;

  const YoloImagePage({
    super.key,
    required this.modelPath,
    this.providers,
    this.providerOptions = const {},
    this.inputSize = 640,
  });

  @override
  State<YoloImagePage> createState() => _YoloImagePageState();
}

class _YoloImagePageState extends State<YoloImagePage> {
  YoloDetector? _detector;
  bool _loading = true;
  String? _error;
  Uint8List? _imageBytes;
  img.Image? _decoded;
  List<Detection> _detections = [];
  double _confThreshold = 0.25;
  final List<String> _sampleImages = [
    'assets/sample_street.png',
    'assets/sample_people.png',
  ];
  int _currentImageIndex = 0;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      _detector = YoloDetector.create(
        widget.modelPath,
        providers: widget.providers,
        providerOptions: widget.providerOptions,
        inputSize: widget.inputSize,
      );
      await _loadSampleImage();
      setState(() => _loading = false);
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _loadSampleImage() async {
    try {
      final bytes = await rootBundle.load(_sampleImages[_currentImageIndex]);
      _imageBytes = bytes.buffer.asUint8List();
      _decoded = img.decodeImage(_imageBytes!);
      if (_decoded == null) throw Exception('Failed to decode image');
      await _runInference();
    } catch (e) {
      setState(() => _error = 'Failed to load image: $e');
    }
  }

  Future<void> _runInference() async {
    if (_detector == null || _decoded == null) return;

    try {
      final sw = Stopwatch()..start();
      final detections = _detector!.detectFromImage(
        _decoded!,
        confThreshold: _confThreshold,
      );
      sw.stop();

      setState(() {
        _detections = detections;
        _error = null;
      });
    } catch (e) {
      setState(() => _error = 'Inference error: $e');
    }
  }

  void _nextImage() {
    _currentImageIndex = (_currentImageIndex + 1) % _sampleImages.length;
    _loadSampleImage();
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO Image Detection (Emulator)'),
        actions: [
          IconButton(
            icon: const Icon(Icons.tune),
            onPressed: _showSettingsSheet,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null && _decoded == null
          ? Center(child: SelectableText(_error!))
          : Column(
              children: [
                Expanded(
                  child: _imageBytes != null
                      ? Stack(
                          children: [
                            Center(
                              child: Image.memory(
                                _imageBytes!,
                                fit: BoxFit.contain,
                              ),
                            ),
                            if (_decoded != null)
                              Positioned.fill(
                                child: CustomPaint(
                                  painter: DetectionPainter(
                                    detections: _detections,
                                    imageWidth: _decoded!.width.toDouble(),
                                    imageHeight: _decoded!.height.toDouble(),
                                  ),
                                ),
                              ),
                          ],
                        )
                      : const Center(child: Text('No image')),
                ),
                // Bottom controls
                Container(
                  padding: const EdgeInsets.all(16),
                  color: Colors.grey[100],
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        'Objects: ${_detections.length}',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 8),
                      ElevatedButton.icon(
                        onPressed: _nextImage,
                        icon: const Icon(Icons.image),
                        label: Text('Sample ${_currentImageIndex + 1}'),
                      ),
                      if (_error != null) ...[
                        const SizedBox(height: 8),
                        SelectableText(
                          _error!,
                          style: const TextStyle(
                            color: Colors.red,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ],
            ),
    );
  }

  void _showSettingsSheet() {
    showModalBottomSheet(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setSheetState) {
            return Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('Confidence Threshold'),
                  Slider(
                    value: _confThreshold,
                    min: 0.1,
                    max: 0.9,
                    divisions: 16,
                    onChanged: (v) {
                      setSheetState(() => _confThreshold = v);
                      setState(() => _confThreshold = v);
                      _runInference();
                    },
                  ),
                  Text('${(_confThreshold * 100).toStringAsFixed(0)}%'),
                ],
              ),
            );
          },
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Detection painter for static images
// ---------------------------------------------------------------------------

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final double imageWidth;
  final double imageHeight;

  DetectionPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

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
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(4)),
        paint,
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
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
