import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

import 'shared.dart';
import 'yolo_camera_page.dart';
import 'yolo_image_page.dart';
import 'yolo_video_page.dart';

enum InputMode { camera, image, video }

/// YOLO model entry for the setup page.
///
/// To add a new model, just add an entry to [YoloModelInfo.available].
class YoloModelInfo {
  final String assetPath;
  final String name;
  final int inputSize;
  final bool isOrt;

  const YoloModelInfo({
    required this.assetPath,
    required this.name,
    required this.inputSize,
    required this.isOrt,
  });

  String get displayName => assetPath.split('/').last;

  String get subtitle {
    final parts = <String>[
      name,
      '${inputSize}x$inputSize',
      isOrt ? 'ORT (optimized)' : 'ONNX (standard)',
    ];
    return parts.join(' • ');
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is YoloModelInfo && other.assetPath == assetPath;

  @override
  int get hashCode => assetPath.hashCode;

  // -----------------------------------------------------------------------
  // Add new models here. Just append a new entry.
  // -----------------------------------------------------------------------
  static const List<YoloModelInfo> available = [
    YoloModelInfo(
      assetPath: 'assets/yolos/yolo11.onnx',
      name: 'YOLOv11',
      inputSize: 640,
      isOrt: false,
    ),
    YoloModelInfo(
      assetPath: 'assets/yolos/yolo11.ort',
      name: 'YOLOv11',
      inputSize: 640,
      isOrt: true,
    ),
    YoloModelInfo(
      assetPath: 'assets/yolos/yolo8n.onnx',
      name: 'YOLOv8n (FP16)',
      inputSize: 320,
      isOrt: false,
    ),
    YoloModelInfo(
      assetPath: 'assets/yolos/yolo8n.ort',
      name: 'YOLOv8n (FP16)',
      inputSize: 320,
      isOrt: true,
    ),
    YoloModelInfo(
      assetPath: 'assets/yolos/yolo5n.onnx',
      name: 'YOLOv5n (FP16)',
      inputSize: 320,
      isOrt: false,
    ),
    YoloModelInfo(
      assetPath: 'assets/yolos/yolo5n.ort',
      name: 'YOLO5n (FP16)',
      inputSize: 320,
      isOrt: true,
    ),
  ];
}

class YoloSetupPage extends StatefulWidget {
  const YoloSetupPage({super.key});

  @override
  State<YoloSetupPage> createState() => _YoloSetupPageState();
}

class _YoloSetupPageState extends State<YoloSetupPage> {
  final bool _isAndroid = Platform.isAndroid;
  final bool _isIOS = Platform.isIOS || Platform.isMacOS;

  // Model discovery
  List<YoloModelInfo> _discoveredModels = [];
  YoloModelInfo? _selectedModelInfo;
  int? _inputSizeOverride;

  // Input mode selection
  InputMode _inputMode = InputMode.image;

  // Provider selection
  List<String> _availableProviders = [];
  final Set<OrtProvider> _selectedProviders = {};
  bool _useAutoProviders = true;
  bool _loading = false;
  String? _error;

  // XNNPACK options (Android)
  int _xnnpackThreads = Platform.numberOfProcessors;

  // NNAPI flags (Android)
  bool _nnapiFp16 = false;
  bool _nnapiNchw = false;
  bool _nnapiCpuDisabled = false;
  bool _nnapiCpuOnly = false;

  // CoreML options (iOS)
  String _coremlComputeUnits = 'ALL';

  @override
  void initState() {
    super.initState();
    _loadProviders();

    _discoveredModels = YoloModelInfo.available;
    // Prefer .ort on Android, .onnx on iOS
    _selectedModelInfo = _discoveredModels.firstWhere(
      (m) => _isAndroid ? m.isOrt : !m.isOrt,
      orElse: () => _discoveredModels.first,
    );
  }

  void _loadProviders() {
    try {
      final runtime = OnnxRuntime.instance;
      final providers = OrtProviders(runtime);
      _availableProviders = providers.getAvailableProviders();

      // Pre-select default providers
      final defaults = providers.getDefaultProvidersForPlatform();
      _selectedProviders.addAll(defaults);

      setState(() {});
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  List<OrtProvider> get _supportedProviders {
    return OrtProvider.values.where((p) {
      return _availableProviders.contains(p.ortName) ||
          // XNNPACK pode aparecer com X minúsculo
          (p == OrtProvider.xnnpack &&
              (_availableProviders.contains('XnnpackExecutionProvider') ||
                  _availableProviders.contains('XNNPACKExecutionProvider'))) ||
          // NNAPI pode aparecer com variações de case
          (p == OrtProvider.nnapi &&
              (_availableProviders.contains('NnapiExecutionProvider') ||
                  _availableProviders.contains('NNAPIExecutionProvider')));
    }).toList();
  }

  Future<void> _startDetection() async {
    if (_selectedModelInfo == null) {
      setState(() => _error = 'No model selected');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final modelPath = await copyYoloModelToTemp(
        asset: _selectedModelInfo!.assetPath,
      );
      final inputSize = _inputSizeOverride ?? _selectedModelInfo!.inputSize;

      final providers = _useAutoProviders ? null : _selectedProviders.toList();

      // Ensure CPU is always last if manual
      if (providers != null && !providers.contains(OrtProvider.cpu)) {
        providers.add(OrtProvider.cpu);
      }

      if (!mounted) return;

      final providerOpts = <OrtProvider, Map<String, String>>{};

      // Android-specific options
      if (_isAndroid) {
        if (_selectedProviders.contains(OrtProvider.xnnpack) ||
            _useAutoProviders) {
          providerOpts[OrtProvider.xnnpack] = XnnpackOptions(
            numThreads: _xnnpackThreads,
          ).toMap();
        }
        if (_selectedProviders.contains(OrtProvider.nnapi) ||
            _useAutoProviders) {
          providerOpts[OrtProvider.nnapi] = {
            if (_nnapiFp16) 'use_fp16': 'true',
            if (_nnapiNchw) 'use_nchw': 'true',
            if (_nnapiCpuDisabled) 'cpu_disabled': 'true',
            if (_nnapiCpuOnly) 'cpu_only': 'true',
          };
        }
      }

      // iOS-specific options
      if (_isIOS) {
        if (_selectedProviders.contains(OrtProvider.coreML) ||
            _useAutoProviders) {
          providerOpts[OrtProvider.coreML] = {
            'MLComputeUnits': _coremlComputeUnits,
          };
        }
      }

      Widget page;
      switch (_inputMode) {
        case InputMode.camera:
          page = YoloCameraPage(
            modelPath: modelPath,
            providers: providers,
            providerOptions: providerOpts,
            inputSize: inputSize,
          );
          break;
        case InputMode.image:
          page = YoloImagePage(
            modelPath: modelPath,
            providers: providers,
            providerOptions: providerOpts,
            inputSize: inputSize,
          );
          break;
        case InputMode.video:
          page = YoloVideoPage(
            modelPath: modelPath,
            providers: providers,
            providerOptions: providerOpts,
            inputSize: inputSize,
          );
          break;
      }

      Navigator.of(context).push(MaterialPageRoute(builder: (_) => page));
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO Setup'),
        backgroundColor: cs.inversePrimary,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Input mode section
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Input Source',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  RadioListTile<InputMode>(
                    title: const Text('Camera'),
                    subtitle: const Text(
                      'Real-time detection using device camera',
                      style: TextStyle(fontSize: 12),
                    ),
                    value: InputMode.camera,
                    groupValue: _inputMode,
                    onChanged: (v) => setState(() => _inputMode = v!),
                    dense: true,
                    secondary: const Icon(Icons.camera_alt),
                  ),
                  RadioListTile<InputMode>(
                    title: const Text('Static Image'),
                    subtitle: const Text(
                      'Test with sample images (works on emulator)',
                      style: TextStyle(fontSize: 12),
                    ),
                    value: InputMode.image,
                    groupValue: _inputMode,
                    onChanged: (v) => setState(() => _inputMode = v!),
                    dense: true,
                    secondary: const Icon(Icons.image),
                  ),
                  RadioListTile<InputMode>(
                    title: const Text('Video File'),
                    subtitle: const Text(
                      'Process video file frame by frame',
                      style: TextStyle(fontSize: 12),
                    ),
                    value: InputMode.video,
                    groupValue: _inputMode,
                    onChanged: (v) => setState(() => _inputMode = v!),
                    dense: true,
                    secondary: const Icon(Icons.video_file),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 12),

          // Model selection section
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Model',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _isAndroid
                        ? 'ORT format is pre-optimized for Android accelerators.'
                        : 'ONNX format works well with CoreML on iOS.',
                    style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                  ),
                  if (_discoveredModels.isEmpty)
                    const Padding(
                      padding: EdgeInsets.only(top: 12),
                      child: Text('No models found in assets/yolos/'),
                    )
                  else
                    ..._discoveredModels.map((model) {
                      return RadioListTile<YoloModelInfo>(
                        title: Text(model.displayName),
                        subtitle: Text(
                          model.subtitle,
                          style: const TextStyle(fontSize: 12),
                        ),
                        value: model,
                        groupValue: _selectedModelInfo,
                        onChanged: (v) =>
                            setState(() => _selectedModelInfo = v),
                        dense: true,
                        secondary: Icon(
                          model.isOrt ? Icons.bolt : Icons.description,
                          color: model.isOrt
                              ? Colors.amber[700]
                              : Colors.grey[600],
                        ),
                      );
                    }),
                ],
              ),
            ),
          ),

          const SizedBox(height: 12),

          // Input size override
          if (_selectedModelInfo != null)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Input Size',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Auto-detected from filename. Override if the model was exported with a different imgsz.',
                      style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                    ),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 8,
                      children: [160, 320, 416, 480, 640].map((size) {
                        final isSelected =
                            (_inputSizeOverride ??
                                _selectedModelInfo!.inputSize) ==
                            size;
                        return ChoiceChip(
                          label: Text('$size'),
                          selected: isSelected,
                          onSelected: (selected) {
                            setState(() {
                              _inputSizeOverride = selected ? size : null;
                            });
                          },
                        );
                      }).toList(),
                    ),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 12),

          // Provider selection section
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Execution Providers',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Available: ${_availableProviders.join(", ")}',
                    style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                  ),
                  const SizedBox(height: 12),
                  SwitchListTile(
                    title: const Text('Auto (platform defaults)'),
                    subtitle: Text(
                      _useAutoProviders
                          ? 'System will choose the best providers'
                          : 'Manual selection below',
                      style: const TextStyle(fontSize: 12),
                    ),
                    value: _useAutoProviders,
                    onChanged: (v) => setState(() => _useAutoProviders = v),
                    dense: true,
                  ),
                  if (!_useAutoProviders) ...[
                    const Divider(),
                    ..._supportedProviders.map((provider) {
                      final isSelected = _selectedProviders.contains(provider);
                      return CheckboxListTile(
                        title: Text(provider.name),
                        subtitle: Text(
                          provider.ortName,
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.grey[600],
                          ),
                        ),
                        value: isSelected,
                        onChanged: provider == OrtProvider.cpu
                            ? null // CPU always included
                            : (v) {
                                setState(() {
                                  if (v == true) {
                                    _selectedProviders.add(provider);
                                  } else {
                                    _selectedProviders.remove(provider);
                                  }
                                });
                              },
                        dense: true,
                        secondary: _providerIcon(provider),
                      );
                    }),
                    // CPU always shown
                    if (!_supportedProviders.contains(OrtProvider.cpu))
                      CheckboxListTile(
                        title: const Text('cpu'),
                        subtitle: Text(
                          OrtProvider.cpu.ortName,
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.grey[600],
                          ),
                        ),
                        value: true,
                        onChanged: null,
                        dense: true,
                        secondary: const Icon(Icons.memory, color: Colors.grey),
                      ),
                  ],
                ],
              ),
            ),
          ),

          const SizedBox(height: 12),

          // === Android-specific options ===
          if (_isAndroid) ...[
            // XNNPACK options
            if (_selectedProviders.contains(OrtProvider.xnnpack) ||
                _useAutoProviders)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'XNNPACK Options',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'XNNPACK uses NEON SIMD for optimized CPU inference on ARM.',
                        style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Threads: $_xnnpackThreads / ${Platform.numberOfProcessors}',
                        style: const TextStyle(fontSize: 13),
                      ),
                      Slider(
                        value: _xnnpackThreads.toDouble(),
                        min: 1,
                        max: Platform.numberOfProcessors.toDouble(),
                        divisions: Platform.numberOfProcessors - 1,
                        label: '$_xnnpackThreads',
                        onChanged: (v) =>
                            setState(() => _xnnpackThreads = v.round()),
                      ),
                      Text(
                        'More threads = faster inference but higher power usage.',
                        style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                      ),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 12),

            // NNAPI options
            if (_selectedProviders.contains(OrtProvider.nnapi) ||
                _useAutoProviders)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'NNAPI Options',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'NNAPI delegates to Android NPU/GPU. Use .ort model for best compatibility.',
                        style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                      ),
                      const SizedBox(height: 8),
                      SwitchListTile(
                        title: const Text('FP16'),
                        subtitle: const Text(
                          'Use half-precision floats for faster inference',
                          style: TextStyle(fontSize: 11),
                        ),
                        value: _nnapiFp16,
                        onChanged: (v) => setState(() => _nnapiFp16 = v),
                        dense: true,
                      ),
                      SwitchListTile(
                        title: const Text('NCHW'),
                        subtitle: const Text(
                          'Use NCHW layout instead of NHWC',
                          style: TextStyle(fontSize: 11),
                        ),
                        value: _nnapiNchw,
                        onChanged: (v) => setState(() => _nnapiNchw = v),
                        dense: true,
                      ),
                      SwitchListTile(
                        title: const Text('CPU Disabled'),
                        subtitle: const Text(
                          'Disable CPU fallback — fail if NPU/GPU can\'t run an op',
                          style: TextStyle(fontSize: 11),
                        ),
                        value: _nnapiCpuDisabled,
                        onChanged: _nnapiCpuOnly
                            ? null
                            : (v) => setState(() => _nnapiCpuDisabled = v),
                        dense: true,
                      ),
                      SwitchListTile(
                        title: const Text('CPU Only'),
                        subtitle: const Text(
                          'Force NNAPI to use CPU only (for testing)',
                          style: TextStyle(fontSize: 11),
                        ),
                        value: _nnapiCpuOnly,
                        onChanged: _nnapiCpuDisabled
                            ? null
                            : (v) => setState(() => _nnapiCpuOnly = v),
                        dense: true,
                      ),
                    ],
                  ),
                ),
              ),
          ],

          // === iOS-specific options ===
          if (_isIOS) ...[
            const SizedBox(height: 12),

            // CoreML options
            if (_selectedProviders.contains(OrtProvider.coreML) ||
                _useAutoProviders)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'CoreML Options',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'CoreML accelerates inference using Apple Neural Engine, GPU, or CPU.',
                        style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Compute Units',
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 8),
                      RadioListTile<String>(
                        title: const Text('ALL'),
                        subtitle: const Text(
                          'Use Neural Engine + GPU + CPU (best performance)',
                          style: TextStyle(fontSize: 11),
                        ),
                        value: 'ALL',
                        groupValue: _coremlComputeUnits,
                        onChanged: (v) =>
                            setState(() => _coremlComputeUnits = v!),
                        dense: true,
                      ),
                      RadioListTile<String>(
                        title: const Text('CPU & Neural Engine'),
                        subtitle: const Text(
                          'Skip GPU, use ANE + CPU',
                          style: TextStyle(fontSize: 11),
                        ),
                        value: 'CPUAndNeuralEngine',
                        groupValue: _coremlComputeUnits,
                        onChanged: (v) =>
                            setState(() => _coremlComputeUnits = v!),
                        dense: true,
                      ),
                      RadioListTile<String>(
                        title: const Text('CPU & GPU'),
                        subtitle: const Text(
                          'Skip Neural Engine, use GPU + CPU',
                          style: TextStyle(fontSize: 11),
                        ),
                        value: 'CPUAndGPU',
                        groupValue: _coremlComputeUnits,
                        onChanged: (v) =>
                            setState(() => _coremlComputeUnits = v!),
                        dense: true,
                      ),
                      RadioListTile<String>(
                        title: const Text('CPU Only'),
                        subtitle: const Text(
                          'CoreML CPU only (for testing/debugging)',
                          style: TextStyle(fontSize: 11),
                        ),
                        value: 'CPUOnly',
                        groupValue: _coremlComputeUnits,
                        onChanged: (v) =>
                            setState(() => _coremlComputeUnits = v!),
                        dense: true,
                      ),
                    ],
                  ),
                ),
              ),
          ],

          const SizedBox(height: 12),

          // Summary
          Card(
            color: cs.primaryContainer.withValues(alpha: 0.3),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Configuration Summary',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  _SummaryRow(
                    'Model',
                    _selectedModelInfo != null
                        ? '${_selectedModelInfo!.displayName} (${_inputSizeOverride ?? _selectedModelInfo!.inputSize}px)'
                        : 'None',
                  ),
                  _SummaryRow(
                    'Providers',
                    _useAutoProviders
                        ? 'Auto (platform defaults)'
                        : [
                            ..._selectedProviders.map((p) => p.name),
                            if (!_selectedProviders.contains(OrtProvider.cpu))
                              'cpu (fallback)',
                          ].join(', '),
                  ),
                  if (_isAndroid) ...[
                    _SummaryRow('XNNPACK', '$_xnnpackThreads threads'),
                    _SummaryRow(
                      'NNAPI',
                      [
                        if (_nnapiFp16) 'FP16',
                        if (_nnapiNchw) 'NCHW',
                        if (_nnapiCpuDisabled) 'CPU disabled',
                        if (_nnapiCpuOnly) 'CPU only',
                        if (!_nnapiFp16 &&
                            !_nnapiNchw &&
                            !_nnapiCpuDisabled &&
                            !_nnapiCpuOnly)
                          'default flags',
                      ].join(', '),
                    ),
                  ],
                  if (_isIOS) _SummaryRow('CoreML', _coremlComputeUnits),
                ],
              ),
            ),
          ),

          if (_error != null) ...[
            const SizedBox(height: 12),
            Card(
              color: cs.errorContainer,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: SelectableText(
                  _error!,
                  style: TextStyle(color: cs.onErrorContainer, fontSize: 13),
                ),
              ),
            ),
          ],

          const SizedBox(height: 24),

          // Start button
          FilledButton.icon(
            onPressed: _loading ? null : _startDetection,
            icon: _loading
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : _getInputIcon(),
            label: Text(_loading ? 'Loading model...' : _getButtonText()),
            style: FilledButton.styleFrom(
              minimumSize: const Size.fromHeight(52),
              textStyle: const TextStyle(fontSize: 16),
            ),
          ),

          const SizedBox(height: 16),
        ],
      ),
    );
  }

  Widget _providerIcon(OrtProvider provider) {
    switch (provider) {
      case OrtProvider.nnapi:
        return Icon(Icons.android, color: Colors.green[700]);
      case OrtProvider.coreML:
        return Icon(Icons.phone_iphone, color: Colors.grey[800]);
      case OrtProvider.xnnpack:
        return Icon(Icons.speed, color: Colors.blue[700]);
      case OrtProvider.qnn:
        return Icon(Icons.developer_board, color: Colors.purple[700]);
      case OrtProvider.cuda:
      case OrtProvider.tensorRT:
        return Icon(Icons.graphic_eq, color: Colors.green[800]);
      case OrtProvider.cpu:
        return const Icon(Icons.memory, color: Colors.grey);
      default:
        return Icon(Icons.extension, color: Colors.orange[700]);
    }
  }

  Icon _getInputIcon() {
    switch (_inputMode) {
      case InputMode.camera:
        return const Icon(Icons.camera_alt);
      case InputMode.image:
        return const Icon(Icons.image);
      case InputMode.video:
        return const Icon(Icons.video_file);
    }
  }

  String _getButtonText() {
    switch (_inputMode) {
      case InputMode.camera:
        return 'Start Camera Detection';
      case InputMode.image:
        return 'Open Image Test';
      case InputMode.video:
        return 'Load Video Detection';
    }
  }
}

class _SummaryRow extends StatelessWidget {
  final String label;
  final String value;

  const _SummaryRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 80,
            child: Text(
              label,
              style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
            ),
          ),
          Expanded(child: Text(value, style: const TextStyle(fontSize: 13))),
        ],
      ),
    );
  }
}
