import 'package:flutter/material.dart';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

import 'shared.dart';

/// Demonstrates the difference between sync (main thread) and isolate
/// inference. A continuously spinning animation makes UI freezes visible.
class IsolatePage extends StatefulWidget {
  final String modelPath;

  const IsolatePage({super.key, required this.modelPath});

  @override
  State<IsolatePage> createState() => _IsolatePageState();
}

class _IsolatePageState extends State<IsolatePage>
    with SingleTickerProviderStateMixin {
  final _runtime = OnnxRuntime.instance;

  // Sync session (main thread)
  OrtSessionWrapper? _syncSession;

  // Isolate session (background thread)
  OrtIsolateSession? _isolateSession;

  bool _loading = true;
  String? _error;

  // Results
  _ComparisonResult? _syncResult;
  _ComparisonResult? _isolateResult;
  bool _runningSync = false;
  bool _runningIsolate = false;

  // Spinner animation — freezes are visible here
  late final AnimationController _spinController;

  // How many consecutive inferences to run (makes freeze more obvious)
  int _iterations = 50;

  @override
  void initState() {
    super.initState();
    _spinController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
    _init();
  }

  Future<void> _init() async {
    try {
      // Sync session
      _syncSession = OrtSessionWrapper.create(widget.modelPath);

      // Isolate session
      _isolateSession = await OrtIsolateSession.create(
        OrtIsolateSessionConfig(modelPath: widget.modelPath),
      );

      setState(() => _loading = false);
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  // ---- Sync inference (blocks main thread) --------------------------------

  void _runSync() {
    setState(() {
      _runningSync = true;
      _syncResult = null;
    });

    final input = syntheticDigit(3);
    final sw = Stopwatch()..start();

    // Run N iterations on the main thread — UI will freeze
    for (var i = 0; i < _iterations; i++) {
      final value =
          OrtValueWrapper.fromFloat(_runtime, [1, 1, 28, 28], input);
      _syncSession!.runFloat(
        {_syncSession!.inputNames.first: value},
        [10],
      );
      value.release();
    }

    sw.stop();

    setState(() {
      _syncResult = _ComparisonResult(
        totalMs: sw.elapsedMicroseconds / 1000.0,
        iterations: _iterations,
        frozeUI: true,
      );
      _runningSync = false;
    });
  }

  // ---- Isolate inference (background thread) ------------------------------

  Future<void> _runIsolate() async {
    setState(() {
      _runningIsolate = true;
      _isolateResult = null;
    });

    try {
      final input = syntheticDigit(3);
      final isolateInput = OrtIsolateInput(
        shape: [1, 1, 28, 28],
        data: input,
      );
      final inputName = _isolateSession!.inputNames.first;

      final sw = Stopwatch()..start();

      // Run N iterations in the background — UI stays smooth
      for (var i = 0; i < _iterations; i++) {
        await _isolateSession!.runFloat(
          {inputName: isolateInput},
          [10],
        );
      }

      sw.stop();

      setState(() {
        _isolateResult = _ComparisonResult(
          totalMs: sw.elapsedMicroseconds / 1000.0,
          iterations: _iterations,
          frozeUI: false,
        );
        _runningIsolate = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _runningIsolate = false;
      });
    }
  }

  @override
  void dispose() {
    _spinController.dispose();
    _syncSession?.dispose();
    _isolateSession?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isRunning = _runningSync || _runningIsolate;

    return Scaffold(
      appBar: AppBar(title: const Text('Isolate vs Sync')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null && _syncSession == null
              ? Center(child: ErrorCard(message: _error!))
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    // -- Spinner indicator --
                    Card(
                      color: cs.tertiaryContainer.withValues(alpha: 0.3),
                      child: Padding(
                        padding: const EdgeInsets.all(20),
                        child: Column(
                          children: [
                            RotationTransition(
                              turns: _spinController,
                              child: Icon(
                                Icons.settings,
                                size: 64,
                                color: cs.tertiary,
                              ),
                            ),
                            const SizedBox(height: 12),
                            Text(
                              'This spinner runs on the UI thread.\n'
                              'If it freezes, the main thread is blocked.',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 13,
                                color: cs.onTertiaryContainer,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),

                    // -- Iteration selector --
                    SectionCard(
                      title: 'Iterations per test',
                      children: [
                        Slider(
                          value: _iterations.toDouble(),
                          min: 10,
                          max: 500,
                          divisions: 49,
                          label: '$_iterations',
                          onChanged: isRunning
                              ? null
                              : (v) =>
                                  setState(() => _iterations = v.round()),
                        ),
                        Center(
                          child: Text(
                            '$_iterations inferences',
                            style: TextStyle(
                              fontSize: 13,
                              color: Colors.grey[600],
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),

                    // -- Buttons --
                    Row(
                      children: [
                        Expanded(
                          child: FilledButton.icon(
                            onPressed: isRunning ? null : _runSync,
                            icon: const Icon(Icons.warning_amber),
                            label: const Text('Sync (freezes)'),
                            style: FilledButton.styleFrom(
                              backgroundColor: cs.error,
                              foregroundColor: cs.onError,
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: FilledButton.icon(
                            onPressed: isRunning ? null : _runIsolate,
                            icon: const Icon(Icons.check_circle_outline),
                            label: const Text('Isolate (smooth)'),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),

                    if (_runningSync)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 8),
                        child: Text(
                          'Running on main thread — watch the spinner freeze!',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: Colors.red,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),

                    if (_runningIsolate)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 8),
                        child: Column(
                          children: [
                            LinearProgressIndicator(),
                            SizedBox(height: 8),
                            Text(
                              'Running in background isolate — spinner stays smooth!',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.green,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),

                    const SizedBox(height: 16),

                    // -- Results --
                    if (_syncResult != null)
                      _ComparisonCard(
                        title: 'Sync (main thread)',
                        result: _syncResult!,
                        color: cs.errorContainer,
                        textColor: cs.onErrorContainer,
                        icon: Icons.warning,
                      ),

                    if (_syncResult != null && _isolateResult != null)
                      const SizedBox(height: 12),

                    if (_isolateResult != null)
                      _ComparisonCard(
                        title: 'Isolate (background)',
                        result: _isolateResult!,
                        color: cs.primaryContainer,
                        textColor: cs.onPrimaryContainer,
                        icon: Icons.check_circle,
                      ),

                    if (_syncResult != null && _isolateResult != null) ...[
                      const SizedBox(height: 12),
                      _VerdictCard(
                        syncResult: _syncResult!,
                        isolateResult: _isolateResult!,
                      ),
                    ],

                    if (_error != null && _syncSession != null) ...[
                      const SizedBox(height: 12),
                      ErrorCard(message: _error!),
                    ],
                  ],
                ),
    );
  }
}

// ---------------------------------------------------------------------------
// Data & widgets
// ---------------------------------------------------------------------------

class _ComparisonResult {
  final double totalMs;
  final int iterations;
  final bool frozeUI;

  double get avgMs => totalMs / iterations;

  const _ComparisonResult({
    required this.totalMs,
    required this.iterations,
    required this.frozeUI,
  });
}

class _ComparisonCard extends StatelessWidget {
  final String title;
  final _ComparisonResult result;
  final Color color;
  final Color textColor;
  final IconData icon;

  const _ComparisonCard({
    required this.title,
    required this.result,
    required this.color,
    required this.textColor,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      color: color.withValues(alpha: 0.5),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: textColor, size: 20),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: textColor,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            InfoRow('Iterations', '${result.iterations}'),
            InfoRow('Total time', '${result.totalMs.toStringAsFixed(1)} ms'),
            InfoRow('Avg / inference', '${result.avgMs.toStringAsFixed(2)} ms'),
            InfoRow(
              'UI frozen?',
              result.frozeUI ? 'YES — spinner stopped' : 'NO — spinner smooth',
              valueColor: result.frozeUI ? Colors.red : Colors.green,
            ),
          ],
        ),
      ),
    );
  }
}

class _VerdictCard extends StatelessWidget {
  final _ComparisonResult syncResult;
  final _ComparisonResult isolateResult;

  const _VerdictCard({
    required this.syncResult,
    required this.isolateResult,
  });

  @override
  Widget build(BuildContext context) {
    final overhead =
        ((isolateResult.totalMs - syncResult.totalMs) / syncResult.totalMs * 100);

    return SectionCard(
      title: 'Verdict',
      children: [
        InfoRow('Sync total', '${syncResult.totalMs.toStringAsFixed(1)} ms'),
        InfoRow(
            'Isolate total', '${isolateResult.totalMs.toStringAsFixed(1)} ms'),
        InfoRow(
          'Isolate overhead',
          '${overhead.toStringAsFixed(1)}%',
          valueColor: overhead > 50 ? Colors.orange : Colors.green,
        ),
        const SizedBox(height: 8),
        const Text(
          'The isolate version has some overhead from message passing, '
          'but the UI stays completely responsive. For heavy models '
          'the trade-off is always worth it.',
          style: TextStyle(fontSize: 12, color: Colors.grey),
        ),
      ],
    );
  }
}
