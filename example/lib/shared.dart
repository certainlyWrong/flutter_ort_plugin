import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

// ---------------------------------------------------------------------------
// Model helper
// ---------------------------------------------------------------------------

Future<String> copyModelToTemp() async {
  final bytes = await rootBundle.load('assets/mnist/mnist-12.onnx');
  final dir = await getTemporaryDirectory();
  final file = File('${dir.path}/mnist-12.onnx');
  await file.writeAsBytes(bytes.buffer.asUint8List());
  return file.path;
}

Future<String> copyYoloModelToTemp({String asset = 'assets/yolo11.ort'}) async {
  final filename = asset.split('/').last;
  final dir = await getTemporaryDirectory();
  final file = File('${dir.path}/$filename');
  if (await file.exists()) return file.path;
  final bytes = await rootBundle.load(asset);
  await file.writeAsBytes(bytes.buffer.asUint8List());
  return file.path;
}

// ---------------------------------------------------------------------------
// Synthetic digit generator
// ---------------------------------------------------------------------------

Float32List syntheticDigit(int digit) {
  final data = Float32List(28 * 28);
  void px(int y, int x) {
    if (y >= 0 && y < 28 && x >= 0 && x < 28) data[y * 28 + x] = 1.0;
  }

  void hLine(int y, int x1, int x2) {
    for (var x = x1; x <= x2; x++) {
      px(y, x);
      px(y + 1, x);
    }
  }

  void vLine(int x, int y1, int y2) {
    for (var y = y1; y <= y2; y++) {
      px(y, x);
      px(y, x + 1);
    }
  }

  switch (digit) {
    case 0:
      hLine(6, 10, 18);
      hLine(22, 10, 18);
      vLine(10, 6, 22);
      vLine(18, 6, 22);
    case 1:
      vLine(14, 6, 22);
      px(8, 13);
      px(7, 12);
      hLine(22, 12, 17);
    case 3:
      hLine(6, 10, 18);
      hLine(14, 10, 18);
      hLine(22, 10, 18);
      vLine(18, 6, 22);
    case 5:
      hLine(6, 10, 18);
      hLine(14, 10, 18);
      hLine(22, 10, 18);
      vLine(10, 6, 14);
      vLine(18, 14, 22);
    case 7:
      hLine(6, 10, 18);
      vLine(18, 6, 14);
      vLine(14, 14, 22);
    default:
      hLine(6, 10, 18);
      hLine(14, 10, 18);
      hLine(22, 10, 18);
      vLine(10, 6, 14);
      vLine(18, 14, 22);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Inference result
// ---------------------------------------------------------------------------

class InferenceResult {
  final int digit;
  final double confidence;
  final List<double> probabilities;
  final double timeMs;

  const InferenceResult({
    required this.digit,
    required this.confidence,
    required this.probabilities,
    required this.timeMs,
  });
}

InferenceResult resultFromLogits(Float32List logits, double timeMs) {
  int predicted = 0;
  double best = logits[0];
  for (var i = 1; i < logits.length; i++) {
    if (logits[i] > best) {
      best = logits[i];
      predicted = i;
    }
  }
  final maxLogit = logits.reduce(max);
  final exps = logits.map((v) => exp(v - maxLogit)).toList();
  final sumExp = exps.reduce((a, b) => a + b);
  final probs = exps.map((e) => e / sumExp).toList();

  return InferenceResult(
    digit: predicted,
    confidence: probs[predicted],
    probabilities: probs,
    timeMs: timeMs,
  );
}

// ---------------------------------------------------------------------------
// Reusable widgets
// ---------------------------------------------------------------------------

class SectionCard extends StatelessWidget {
  final String title;
  final List<Widget> children;

  const SectionCard({super.key, required this.title, required this.children});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: Theme.of(
                context,
              ).textTheme.titleSmall?.copyWith(color: Colors.grey[600]),
            ),
            const SizedBox(height: 8),
            ...children,
          ],
        ),
      ),
    );
  }
}

class InfoRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;

  const InfoRow(this.label, this.value, {super.key, this.valueColor});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 140,
            child: Text(
              label,
              style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: TextStyle(fontSize: 13, color: valueColor),
            ),
          ),
        ],
      ),
    );
  }
}

class ResultCard extends StatelessWidget {
  final String title;
  final InferenceResult result;

  const ResultCard({super.key, required this.title, required this.result});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      color: cs.primaryContainer.withValues(alpha: 0.4),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: Theme.of(
                context,
              ).textTheme.titleSmall?.copyWith(color: cs.onPrimaryContainer),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Text(
                  '${result.digit}',
                  style: TextStyle(
                    fontSize: 48,
                    fontWeight: FontWeight.bold,
                    color: cs.primary,
                  ),
                ),
                const SizedBox(width: 20),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(fontSize: 14),
                    ),
                    Text(
                      'Time: ${result.timeMs.toStringAsFixed(2)} ms',
                      style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 12),
            for (var i = 0; i < result.probabilities.length; i++)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 1),
                child: Row(
                  children: [
                    SizedBox(
                      width: 16,
                      child: Text(
                        '$i',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: i == result.digit
                              ? FontWeight.bold
                              : FontWeight.normal,
                        ),
                      ),
                    ),
                    const SizedBox(width: 4),
                    Expanded(
                      child: LinearProgressIndicator(
                        value: result.probabilities[i].clamp(0.0, 1.0),
                        backgroundColor: Colors.grey[300],
                        color: i == result.digit ? cs.primary : cs.outline,
                        minHeight: 8,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                    const SizedBox(width: 6),
                    SizedBox(
                      width: 42,
                      child: Text(
                        '${(result.probabilities[i] * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(fontSize: 10),
                        textAlign: TextAlign.right,
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class ErrorCard extends StatelessWidget {
  final String message;

  const ErrorCard({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Theme.of(context).colorScheme.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SelectableText(
          message,
          style: TextStyle(
            color: Theme.of(context).colorScheme.onErrorContainer,
          ),
        ),
      ),
    );
  }
}
