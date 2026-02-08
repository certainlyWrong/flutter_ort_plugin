import 'package:flutter/material.dart';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

import 'shared.dart';

class BenchmarkPage extends StatefulWidget {
  final String modelPath;

  const BenchmarkPage({super.key, required this.modelPath});

  @override
  State<BenchmarkPage> createState() => _BenchmarkPageState();
}

class _BenchmarkPageState extends State<BenchmarkPage> {
  final _runtime = OnnxRuntime.instance;
  OrtSessionWrapper? _session;
  bool _loading = true;
  String? _error;
  bool _running = false;

  int _runs = 50;
  List<double>? _times;

  @override
  void initState() {
    super.initState();
    _init();
  }

  void _init() {
    try {
      _session = OrtSessionWrapper.create(widget.modelPath);
      setState(() => _loading = false);
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _runBenchmark() {
    setState(() {
      _running = true;
      _times = null;
    });

    try {
      final input = syntheticDigit(3);
      final times = <double>[];

      // Warmup
      final warmup = OrtValueWrapper.fromFloat(_runtime, [1, 1, 28, 28], input);
      _session!.runFloat({_session!.inputNames.first: warmup}, [10]);
      warmup.release();

      for (var i = 0; i < _runs; i++) {
        final sw = Stopwatch()..start();
        final value =
            OrtValueWrapper.fromFloat(_runtime, [1, 1, 28, 28], input);
        _session!.runFloat({_session!.inputNames.first: value}, [10]);
        value.release();
        sw.stop();
        times.add(sw.elapsedMicroseconds / 1000.0);
      }

      setState(() {
        _times = times;
        _running = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _running = false;
      });
    }
  }

  @override
  void dispose() {
    _session?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Benchmark')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null && _session == null
              ? Center(child: ErrorCard(message: _error!))
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    // Providers info
                    SectionCard(
                      title: 'Session providers',
                      children: [
                        InfoRow(
                          'Default',
                          OrtProviders(_runtime)
                              .getDefaultProvidersForPlatform()
                              .map((p) => p.name)
                              .join(', '),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),

                    // Runs selector
                    SectionCard(
                      title: 'Configuration',
                      children: [
                        Slider(
                          value: _runs.toDouble(),
                          min: 10,
                          max: 200,
                          divisions: 19,
                          label: '$_runs',
                          onChanged: _running
                              ? null
                              : (v) => setState(() => _runs = v.round()),
                        ),
                        Center(
                          child: Text(
                            '$_runs runs',
                            style: TextStyle(
                                fontSize: 13, color: Colors.grey[600]),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),

                    FilledButton.icon(
                      onPressed: _running ? null : _runBenchmark,
                      icon: _running
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child:
                                  CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.speed),
                      label: Text(_running ? 'Running...' : 'Run Benchmark'),
                    ),
                    const SizedBox(height: 16),

                    if (_times != null) _StatsCard(times: _times!),

                    if (_error != null && _session != null) ...[
                      const SizedBox(height: 12),
                      ErrorCard(message: _error!),
                    ],
                  ],
                ),
    );
  }
}

class _StatsCard extends StatelessWidget {
  final List<double> times;

  const _StatsCard({required this.times});

  @override
  Widget build(BuildContext context) {
    final sorted = List<double>.from(times)..sort();
    final avg = times.reduce((a, b) => a + b) / times.length;
    final median = sorted[sorted.length ~/ 2];
    final best = sorted.first;
    final worst = sorted.last;
    final p95 = sorted[(sorted.length * 0.95).floor()];
    final p99 = sorted[(sorted.length * 0.99).floor()];
    final stdDev = _stdDev(times, avg);

    return Column(
      children: [
        SectionCard(
          title: 'Results — ${times.length} runs',
          children: [
            InfoRow('Average', '${avg.toStringAsFixed(2)} ms'),
            InfoRow('Median', '${median.toStringAsFixed(2)} ms'),
            InfoRow('Best', '${best.toStringAsFixed(2)} ms'),
            InfoRow('Worst', '${worst.toStringAsFixed(2)} ms'),
            InfoRow('P95', '${p95.toStringAsFixed(2)} ms'),
            InfoRow('P99', '${p99.toStringAsFixed(2)} ms'),
            InfoRow('Std dev', '${stdDev.toStringAsFixed(2)} ms'),
          ],
        ),
        const SizedBox(height: 12),
        SectionCard(
          title: 'Distribution',
          children: [
            SizedBox(
              height: 120,
              child: CustomPaint(
                size: const Size(double.infinity, 120),
                painter: _HistogramPainter(
                  times: times,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  static double _stdDev(List<double> values, double mean) {
    final sumSq = values.fold<double>(
        0, (sum, v) => sum + (v - mean) * (v - mean));
    return (sumSq / values.length).clamp(0, double.infinity);
  }
}

class _HistogramPainter extends CustomPainter {
  final List<double> times;
  final Color color;

  _HistogramPainter({required this.times, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    if (times.isEmpty) return;

    const bins = 20;
    final sorted = List<double>.from(times)..sort();
    final minV = sorted.first;
    final maxV = sorted.last;
    final range = maxV - minV;
    if (range == 0) return;

    final binWidth = range / bins;
    final counts = List<int>.filled(bins, 0);
    for (final t in times) {
      var idx = ((t - minV) / binWidth).floor();
      if (idx >= bins) idx = bins - 1;
      counts[idx]++;
    }

    final maxCount = counts.reduce((a, b) => a > b ? a : b);
    if (maxCount == 0) return;

    final barW = size.width / bins;
    final paint = Paint()..color = color.withValues(alpha: 0.7);

    for (var i = 0; i < bins; i++) {
      final barH = (counts[i] / maxCount) * size.height;
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(
            i * barW + 1,
            size.height - barH,
            barW - 2,
            barH,
          ),
          const Radius.circular(2),
        ),
        paint,
      );
    }

    // Axis labels
    final textStyle = TextStyle(fontSize: 9, color: color.withValues(alpha: 0.6));
    final minPainter = TextPainter(
      text: TextSpan(text: '${minV.toStringAsFixed(1)}ms', style: textStyle),
      textDirection: TextDirection.ltr,
    )..layout();
    minPainter.paint(canvas, Offset(0, size.height - minPainter.height));

    final maxPainter = TextPainter(
      text: TextSpan(text: '${maxV.toStringAsFixed(1)}ms', style: textStyle),
      textDirection: TextDirection.ltr,
    )..layout();
    maxPainter.paint(
      canvas,
      Offset(size.width - maxPainter.width, size.height - maxPainter.height),
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
