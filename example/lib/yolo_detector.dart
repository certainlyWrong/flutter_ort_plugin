import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';
import 'package:image/image.dart' as img;

// ---------------------------------------------------------------------------
// COCO 80-class labels
// ---------------------------------------------------------------------------

const List<String> cocoLabels = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
  'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
  'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
  'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush',
];

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------

class Detection {
  final double x1, y1, x2, y2;
  final int classId;
  final String label;
  final double confidence;

  const Detection({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.classId,
    required this.label,
    required this.confidence,
  });
}

// ---------------------------------------------------------------------------
// YoloDetector — wraps OrtSessionWrapper for YOLOv11
// ---------------------------------------------------------------------------

class YoloDetector {
  static const int inputSize = 640;
  static const int numClasses = 80;
  static const int numBoxes = 8400;

  final OrtSessionWrapper _session;
  final OnnxRuntime _runtime;

  YoloDetector._(this._session, this._runtime);

  static YoloDetector create(String modelPath) {
    final runtime = OnnxRuntime.instance;
    final session = OrtSessionWrapper.create(modelPath);
    return YoloDetector._(session, runtime);
  }

  void dispose() {
    _session.dispose();
  }

  /// Run detection on a CameraImage (YUV420 from Android / BGRA from iOS).
  /// Returns detections in normalized [0,1] coordinates relative to the
  /// original image.
  List<Detection> detectFromCameraImage(
    CameraImage cameraImage, {
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) {
    final rgbImage = _cameraImageToImage(cameraImage);
    return detectFromImage(
      rgbImage,
      confThreshold: confThreshold,
      iouThreshold: iouThreshold,
    );
  }

  /// Run detection on an [img.Image].
  List<Detection> detectFromImage(
    img.Image image, {
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) {
    final preprocessed = _preprocess(image);
    final inputValue = OrtValueWrapper.fromFloat(
      _runtime,
      [1, 3, inputSize, inputSize],
      preprocessed,
    );

    try {
      // Output shape: (1, 84, 8400) → 84 * 8400 = 705600
      final outputs = _session.runFloat(
        {_session.inputNames.first: inputValue},
        [numBoxes * (4 + numClasses)],
      );

      final rawOutput = outputs.first;
      return _postprocess(
        rawOutput,
        image.width,
        image.height,
        confThreshold,
        iouThreshold,
      );
    } finally {
      inputValue.release();
    }
  }

  // -------------------------------------------------------------------------
  // Preprocessing: resize with letterbox, normalize to [0,1], CHW layout
  // -------------------------------------------------------------------------

  Float32List _preprocess(img.Image source) {
    // Letterbox resize
    final scale = min(
      inputSize / source.width,
      inputSize / source.height,
    );
    final newW = (source.width * scale).round();
    final newH = (source.height * scale).round();
    final padX = (inputSize - newW) ~/ 2;
    final padY = (inputSize - newH) ~/ 2;

    final resized = img.copyResize(source, width: newW, height: newH);

    final data = Float32List(3 * inputSize * inputSize);
    // Fill with 114/255 (YOLO gray padding)
    final padVal = 114.0 / 255.0;
    for (var i = 0; i < data.length; i++) {
      data[i] = padVal;
    }

    // Copy resized image into the center (CHW, normalized)
    final chSize = inputSize * inputSize;
    for (var y = 0; y < newH; y++) {
      for (var x = 0; x < newW; x++) {
        final pixel = resized.getPixel(x, y);
        final outY = y + padY;
        final outX = x + padX;
        data[0 * chSize + outY * inputSize + outX] = pixel.r / 255.0;
        data[1 * chSize + outY * inputSize + outX] = pixel.g / 255.0;
        data[2 * chSize + outY * inputSize + outX] = pixel.b / 255.0;
      }
    }

    return data;
  }

  // -------------------------------------------------------------------------
  // Postprocessing: parse (1, 84, 8400) → detections with NMS
  // -------------------------------------------------------------------------

  List<Detection> _postprocess(
    Float32List output,
    int origW,
    int origH,
    double confThreshold,
    double iouThreshold,
  ) {
    // output is (1, 84, 8400) flattened row-major:
    //   row 0..3  = cx, cy, w, h  (in 640x640 space)
    //   row 4..83 = class scores
    // Index: output[row * 8400 + col]

    final scale = min(inputSize / origW, inputSize / origH);
    final padX = (inputSize - origW * scale) / 2;
    final padY = (inputSize - origH * scale) / 2;

    final candidates = <Detection>[];

    for (var i = 0; i < numBoxes; i++) {
      // Find best class
      int bestClass = 0;
      double bestScore = output[4 * numBoxes + i];
      for (var c = 1; c < numClasses; c++) {
        final score = output[(4 + c) * numBoxes + i];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < confThreshold) continue;

      // Decode box (cx, cy, w, h) in 640x640 letterboxed space
      final cx = output[0 * numBoxes + i];
      final cy = output[1 * numBoxes + i];
      final w = output[2 * numBoxes + i];
      final h = output[3 * numBoxes + i];

      // Convert to original image coordinates
      final x1 = ((cx - w / 2) - padX) / scale;
      final y1 = ((cy - h / 2) - padY) / scale;
      final x2 = ((cx + w / 2) - padX) / scale;
      final y2 = ((cy + h / 2) - padY) / scale;

      // Clamp to image bounds
      final cx1 = x1.clamp(0.0, origW.toDouble());
      final cy1 = y1.clamp(0.0, origH.toDouble());
      final cx2 = x2.clamp(0.0, origW.toDouble());
      final cy2 = y2.clamp(0.0, origH.toDouble());

      // Normalize to [0, 1]
      candidates.add(Detection(
        x1: cx1 / origW,
        y1: cy1 / origH,
        x2: cx2 / origW,
        y2: cy2 / origH,
        classId: bestClass,
        label: bestClass < cocoLabels.length
            ? cocoLabels[bestClass]
            : 'class_$bestClass',
        confidence: bestScore,
      ));
    }

    // NMS
    return _nms(candidates, iouThreshold);
  }

  List<Detection> _nms(List<Detection> detections, double iouThreshold) {
    // Sort by confidence descending
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    final kept = <Detection>[];
    final suppressed = List.filled(detections.length, false);

    for (var i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;
      kept.add(detections[i]);

      for (var j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;
        if (detections[i].classId != detections[j].classId) continue;
        if (_iou(detections[i], detections[j]) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return kept;
  }

  double _iou(Detection a, Detection b) {
    final interX1 = max(a.x1, b.x1);
    final interY1 = max(a.y1, b.y1);
    final interX2 = min(a.x2, b.x2);
    final interY2 = min(a.y2, b.y2);

    final interW = max(0.0, interX2 - interX1);
    final interH = max(0.0, interY2 - interY1);
    final interArea = interW * interH;

    final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    return interArea / (areaA + areaB - interArea + 1e-6);
  }

  // -------------------------------------------------------------------------
  // Camera image conversion
  // -------------------------------------------------------------------------

  img.Image _cameraImageToImage(CameraImage cameraImage) {
    if (cameraImage.format.group == ImageFormatGroup.yuv420) {
      return _yuv420ToImage(cameraImage);
    } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
      return _bgra8888ToImage(cameraImage);
    } else {
      throw UnsupportedError(
        'Unsupported camera image format: ${cameraImage.format.group}',
      );
    }
  }

  img.Image _yuv420ToImage(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;
    final yPlane = cameraImage.planes[0];
    final uPlane = cameraImage.planes[1];
    final vPlane = cameraImage.planes[2];

    final image = img.Image(width: width, height: height);

    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final yIndex = y * yRowStride + x;
        final uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;

        final yVal = yPlane.bytes[yIndex];
        final uVal = uPlane.bytes[uvIndex];
        final vVal = vPlane.bytes[uvIndex];

        final r = (yVal + 1.370705 * (vVal - 128)).round().clamp(0, 255);
        final g = (yVal - 0.337633 * (uVal - 128) - 0.698001 * (vVal - 128))
            .round()
            .clamp(0, 255);
        final b = (yVal + 1.732446 * (uVal - 128)).round().clamp(0, 255);

        image.setPixelRgb(x, y, r, g, b);
      }
    }

    return image;
  }

  img.Image _bgra8888ToImage(CameraImage cameraImage) {
    final plane = cameraImage.planes[0];
    final width = cameraImage.width;
    final height = cameraImage.height;
    final image = img.Image(width: width, height: height);

    final bytes = plane.bytes;
    final rowStride = plane.bytesPerRow;

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final idx = y * rowStride + x * 4;
        final b = bytes[idx];
        final g = bytes[idx + 1];
        final r = bytes[idx + 2];
        image.setPixelRgb(x, y, r, g, b);
      }
    }

    return image;
  }
}
