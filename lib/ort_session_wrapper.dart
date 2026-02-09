import 'dart:ffi';
import 'dart:typed_data';

import 'bindings/onnxruntime_generated.dart';
import 'onnx_runtinme.dart';
import 'ort_providers.dart';
import 'ort_value_wrapper.dart';
import 'session_config.dart';

class OrtSessionWrapper {
  final OnnxRuntime _runtime;
  late final Pointer<OrtSession> _session;
  late final Pointer<OrtSessionOptions> _sessionOptions;
  late final List<String> _inputNames;
  late final List<String> _outputNames;
  bool _disposed = false;

  OrtSessionWrapper._(this._runtime);

  /// Creates a session with **automatic** provider selection based on the
  /// current platform. Providers are chosen via
  /// [OrtProviders.getDefaultProvidersForPlatform].
  ///
  /// Use [providerOptions] to pass per-provider configuration, e.g.:
  /// ```dart
  /// OrtSessionWrapper.create(
  ///   modelPath,
  ///   providerOptions: {
  ///     OrtProvider.coreML: {'MLComputeUnits': 'ALL'},
  ///   },
  /// );
  /// ```
  static OrtSessionWrapper create(
    String modelPath, {
    OnnxRuntime? runtime,
    Map<OrtProvider, Map<String, String>> providerOptions = const {},
    SessionConfig sessionConfig = const SessionConfig(),
  }) {
    final rt = runtime ?? OnnxRuntime.instance;
    rt.ensureInitialized();

    final wrapper = OrtSessionWrapper._(rt);
    wrapper._sessionOptions = rt.createSessionOptions();
    _applySessionConfig(rt, wrapper._sessionOptions, sessionConfig);

    final providers = OrtProviders(rt);
    providers.appendDefaultProviders(
      wrapper._sessionOptions,
      providerOptions: providerOptions,
    );

    wrapper._session = rt.createSession(modelPath, wrapper._sessionOptions);
    wrapper._inputNames = rt.getSessionInputNames(wrapper._session);
    wrapper._outputNames = rt.getSessionOutputNames(wrapper._session);

    return wrapper;
  }

  /// Creates a session with **manual** provider selection.
  ///
  /// ```dart
  /// OrtSessionWrapper.createWithProviders(
  ///   modelPath,
  ///   providers: [OrtProvider.coreML, OrtProvider.cpu],
  ///   providerOptions: {
  ///     OrtProvider.coreML: {'MLComputeUnits': 'CPUAndGPU'},
  ///   },
  /// );
  /// ```
  static OrtSessionWrapper createWithProviders(
    String modelPath, {
    required List<OrtProvider> providers,
    OnnxRuntime? runtime,
    Map<OrtProvider, Map<String, String>> providerOptions = const {},
    SessionConfig sessionConfig = const SessionConfig(),
  }) {
    final rt = runtime ?? OnnxRuntime.instance;
    rt.ensureInitialized();

    final wrapper = OrtSessionWrapper._(rt);
    wrapper._sessionOptions = rt.createSessionOptions();
    _applySessionConfig(rt, wrapper._sessionOptions, sessionConfig);

    final ortProviders = OrtProviders(rt);
    ortProviders.appendProviders(
      wrapper._sessionOptions,
      providers,
      providerOptions: providerOptions,
    );

    wrapper._session = rt.createSession(modelPath, wrapper._sessionOptions);
    wrapper._inputNames = rt.getSessionInputNames(wrapper._session);
    wrapper._outputNames = rt.getSessionOutputNames(wrapper._session);

    return wrapper;
  }

  /// Creates a session with a **custom** configuration callback for full
  /// control over session options and providers.
  static OrtSessionWrapper createWithCustomConfig(
    String modelPath, {
    OnnxRuntime? runtime,
    required void Function(
      OrtProviders providers,
      Pointer<OrtSessionOptions> options,
    )
    configure,
  }) {
    final rt = runtime ?? OnnxRuntime.instance;
    rt.ensureInitialized();

    final wrapper = OrtSessionWrapper._(rt);
    wrapper._sessionOptions = rt.createSessionOptions();

    final providers = OrtProviders(rt);
    configure(providers, wrapper._sessionOptions);

    wrapper._session = rt.createSession(modelPath, wrapper._sessionOptions);
    wrapper._inputNames = rt.getSessionInputNames(wrapper._session);
    wrapper._outputNames = rt.getSessionOutputNames(wrapper._session);

    return wrapper;
  }

  List<String> get inputNames {
    _ensureNotDisposed();
    return List.unmodifiable(_inputNames);
  }

  List<String> get outputNames {
    _ensureNotDisposed();
    return List.unmodifiable(_outputNames);
  }

  int get inputCount => _inputNames.length;
  int get outputCount => _outputNames.length;

  List<OrtValueWrapper> run(Map<String, OrtValueWrapper> inputs) {
    _ensureNotDisposed();

    final orderedInputNames = <String>[];
    final orderedInputPtrs = <Pointer<OrtValue>>[];

    for (final name in _inputNames) {
      if (!inputs.containsKey(name)) {
        throw ArgumentError(
          'Missing input: "$name". '
          'Expected inputs: ${_inputNames.join(", ")}',
        );
      }
      orderedInputNames.add(name);
      orderedInputPtrs.add(inputs[name]!.pointer);
    }

    final outputPtrs = _runtime.run(
      _session,
      inputNames: orderedInputNames,
      inputValues: orderedInputPtrs,
      outputNames: _outputNames,
    );

    return outputPtrs
        .map((ptr) => OrtValueWrapper.fromPointer(_runtime, ptr))
        .toList();
  }

  List<Float32List> runFloat(
    Map<String, OrtValueWrapper> inputs,
    List<int> outputElementCounts,
  ) {
    _ensureNotDisposed();

    if (outputElementCounts.length != _outputNames.length) {
      throw ArgumentError(
        'outputElementCounts length (${outputElementCounts.length}) '
        'must match output count (${_outputNames.length}).',
      );
    }

    final outputs = run(inputs);
    final results = <Float32List>[];

    try {
      for (var i = 0; i < outputs.length; i++) {
        results.add(outputs[i].toFloatList(outputElementCounts[i]));
      }
    } finally {
      for (final output in outputs) {
        output.release();
      }
    }

    return results;
  }

  void dispose() {
    if (_disposed) return;
    _runtime.releaseSession(_session);
    _runtime.releaseSessionOptions(_sessionOptions);
    _disposed = true;
  }

  bool get isDisposed => _disposed;

  void _ensureNotDisposed() {
    if (_disposed) {
      throw StateError('OrtSessionWrapper has already been disposed.');
    }
  }

  static void _applySessionConfig(
    OnnxRuntime rt,
    Pointer<OrtSessionOptions> options,
    SessionConfig config,
  ) {
    if (config.intraOpThreads > 0) {
      rt.setIntraOpNumThreads(options, config.intraOpThreads);
    }
    if (config.interOpThreads > 0) {
      rt.setInterOpNumThreads(options, config.interOpThreads);
    }
    rt.setGraphOptimizationLevel(options, config.graphOptimizationLevel.value);
    rt.setExecutionMode(options, config.executionMode.value);
  }
}
