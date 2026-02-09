import 'package:flutter/material.dart';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

import 'shared.dart';

class ProvidersPage extends StatefulWidget {
  final String modelPath;

  const ProvidersPage({super.key, required this.modelPath});

  @override
  State<ProvidersPage> createState() => _ProvidersPageState();
}

class _ProvidersPageState extends State<ProvidersPage> {
  final _runtime = OnnxRuntime.instance;
  late final OrtProviders _providers;
  late final List<String> _available;
  late final List<OrtProvider> _defaults;

  // Manual selection
  final Set<OrtProvider> _selected = {OrtProvider.cpu};
  OrtSessionWrapper? _manualSession;
  InferenceResult? _result;
  String? _error;
  bool _running = false;

  @override
  void initState() {
    super.initState();
    _providers = OrtProviders(_runtime);
    _available = _providers.getAvailableProviders();
    _defaults = _providers.getDefaultProvidersForPlatform();
  }

  void _createAndRun() {
    setState(() {
      _running = true;
      _result = null;
      _error = null;
    });

    try {
      _manualSession?.dispose();
      _manualSession = OrtSessionWrapper.createWithProviders(
        widget.modelPath,
        providers: _selected.toList(),
      );

      final input = syntheticDigit(7);
      final sw = Stopwatch()..start();

      final value = OrtValueWrapper.fromFloat(_runtime, [1, 1, 28, 28], input);
      final outputs = _manualSession!.runFloat(
        {_manualSession!.inputNames.first: value},
        [10],
      );
      value.release();
      sw.stop();

      setState(() {
        _result =
            resultFromLogits(outputs.first, sw.elapsedMicroseconds / 1000.0);
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
    _manualSession?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Execution Providers')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Available providers from runtime
          SectionCard(
            title: 'Available providers (from ORT runtime)',
            children: [
              for (final name in _available)
                InfoRow('', name, valueColor: Colors.green[700]),
            ],
          ),
          const SizedBox(height: 12),

          // Default per platform
          SectionCard(
            title: 'Default for this platform',
            children: [
              for (final p in _defaults)
                InfoRow(p.name, p.ortName),
            ],
          ),
          const SizedBox(height: 12),

          // Per-enum availability
          SectionCard(
            title: 'OrtProvider enum check',
            children: [
              for (final p in OrtProvider.values)
                InfoRow(
                  p.name,
                  _providers.isProviderAvailable(p) ? 'Available' : 'Not available',
                  valueColor: _providers.isProviderAvailable(p)
                      ? Colors.green
                      : Colors.grey,
                ),
            ],
          ),
          const SizedBox(height: 24),

          // Manual selection
          SectionCard(
            title: 'Manual provider selection',
            children: [
              const Text(
                'Select providers and run inference:',
                style: TextStyle(fontSize: 13),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 4,
                children: [
                  for (final p in OrtProvider.values)
                    FilterChip(
                      label: Text(p.name),
                      selected: _selected.contains(p),
                      onSelected: (on) {
                        setState(() {
                          if (on) {
                            _selected.add(p);
                          } else if (p != OrtProvider.cpu) {
                            _selected.remove(p);
                          }
                        });
                      },
                    ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                'Selected: ${_selected.map((p) => p.name).join(", ")}',
                style: TextStyle(fontSize: 12, color: Colors.grey[600]),
              ),
            ],
          ),
          const SizedBox(height: 16),

          FilledButton.icon(
            onPressed: _running ? null : _createAndRun,
            icon: const Icon(Icons.play_arrow),
            label: Text(_running
                ? 'Running...'
                : 'Create session & run'),
          ),
          const SizedBox(height: 16),

          if (_result != null)
            ResultCard(
              title: 'Result (${_selected.map((p) => p.name).join(", ")})',
              result: _result!,
            ),

          if (_error != null) ...[
            const SizedBox(height: 12),
            ErrorCard(message: _error!),
          ],
        ],
      ),
    );
  }
}
