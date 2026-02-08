import 'package:flutter/material.dart';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

import 'shared.dart';

class InferencePage extends StatefulWidget {
  final String modelPath;

  const InferencePage({super.key, required this.modelPath});

  @override
  State<InferencePage> createState() => _InferencePageState();
}

class _InferencePageState extends State<InferencePage> {
  final _runtime = OnnxRuntime.instance;
  OrtSessionWrapper? _session;
  bool _loading = true;
  String? _error;
  InferenceResult? _result;
  bool _running = false;
  int _selectedDigit = 7;

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

  void _run() {
    setState(() {
      _running = true;
      _result = null;
    });

    try {
      final input = syntheticDigit(_selectedDigit);
      final sw = Stopwatch()..start();

      final value = OrtValueWrapper.fromFloat(_runtime, [1, 1, 28, 28], input);
      final outputs = _session!.runFloat(
        {_session!.inputNames.first: value},
        [10],
      );
      value.release();
      sw.stop();

      setState(() {
        _result = resultFromLogits(outputs.first, sw.elapsedMicroseconds / 1000.0);
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
      appBar: AppBar(title: const Text('Basic Inference')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null && _session == null
              ? Center(child: ErrorCard(message: _error!))
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    SectionCard(
                      title: 'Session info',
                      children: [
                        InfoRow('Inputs', _session!.inputNames.join(', ')),
                        InfoRow('Outputs', _session!.outputNames.join(', ')),
                        InfoRow('Input shape', '[1, 1, 28, 28]'),
                        InfoRow('Output shape', '[1, 10]  (digits 0-9)'),
                      ],
                    ),
                    const SizedBox(height: 16),

                    // Digit selector
                    SectionCard(
                      title: 'Select digit to classify',
                      children: [
                        Wrap(
                          spacing: 8,
                          children: [
                            for (final d in [0, 1, 3, 5, 7])
                              ChoiceChip(
                                label: Text('$d'),
                                selected: _selectedDigit == d,
                                onSelected: (_) =>
                                    setState(() => _selectedDigit = d),
                              ),
                          ],
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),

                    FilledButton.icon(
                      onPressed: _running ? null : _run,
                      icon: const Icon(Icons.play_arrow),
                      label: Text(_running ? 'Running...' : 'Run Inference'),
                    ),
                    const SizedBox(height: 16),

                    if (_result != null)
                      ResultCard(
                        title: 'Result (auto providers)',
                        result: _result!,
                      ),

                    if (_error != null && _session != null) ...[
                      const SizedBox(height: 12),
                      ErrorCard(message: _error!),
                    ],
                  ],
                ),
    );
  }
}
