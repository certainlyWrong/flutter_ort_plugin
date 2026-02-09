import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'onnx_runtinme.dart';
import 'ort_providers.dart';
import 'ort_session_wrapper.dart';
import 'ort_value_wrapper.dart';

/// Configuration passed to the background isolate to recreate the session.
class OrtIsolateSessionConfig {
  final String modelPath;
  final String? libraryPath;
  final int logLevel;
  final String logId;
  final List<OrtProvider>? providers;
  final Map<OrtProvider, Map<String, String>> providerOptions;

  const OrtIsolateSessionConfig({
    required this.modelPath,
    this.libraryPath,
    this.logLevel = 3,
    this.logId = 'OrtIsolate',
    this.providers,
    this.providerOptions = const {},
  });
}

/// Input data for [OrtIsolateSession.runFloat].
///
/// A plain Dart object that can safely cross isolate boundaries (unlike
/// [OrtValueWrapper] which holds a native pointer).
class OrtIsolateInput {
  final List<int> shape;
  final Float32List data;

  const OrtIsolateInput({required this.shape, required this.data});
}

/// An ONNX Runtime session that runs entirely in a background [Isolate].
///
/// All heavy operations (model loading, inference) happen off the main thread,
/// so the Flutter UI never freezes.
///
/// ```dart
/// final session = await OrtIsolateSession.create(
///   OrtIsolateSessionConfig(modelPath: 'path/to/model.onnx'),
/// );
///
/// final results = await session.runFloat(
///   {'input': OrtIsolateInput(shape: [1, 1, 28, 28], data: inputData)},
///   [10],
/// );
///
/// await session.dispose();
/// ```
class OrtIsolateSession {
  final Isolate _isolate;
  final SendPort _workerPort;
  final Stream<dynamic> _responses;
  final List<String> _inputNames;
  final List<String> _outputNames;
  bool _disposed = false;

  OrtIsolateSession._({
    required Isolate isolate,
    required SendPort workerPort,
    required Stream<dynamic> responses,
    required List<String> inputNames,
    required List<String> outputNames,
  })  : _isolate = isolate,
        _workerPort = workerPort,
        _responses = responses,
        _inputNames = inputNames,
        _outputNames = outputNames;

  /// Creates a session in a background isolate.
  ///
  /// If [OrtIsolateSessionConfig.providers] is set, those providers are used.
  /// Otherwise, providers are auto-selected for the current platform.
  ///
  /// The model is loaded inside the isolate — this call is fully async and
  /// will **not** block the UI thread.
  static Future<OrtIsolateSession> create(
    OrtIsolateSessionConfig config,
  ) async {
    final receivePort = ReceivePort();
    final broadcastStream = receivePort.asBroadcastStream();

    final isolate = await Isolate.spawn(
      _workerEntryPoint,
      _WorkerInit(sendPort: receivePort.sendPort, config: config),
    );

    // Phase 1: worker sends its SendPort
    final workerPort = await broadcastStream.first as SendPort;

    // Phase 2: worker sends _ReadyResponse or _ErrorResponse
    final readyMsg = await broadcastStream.first;
    if (readyMsg is _ErrorResponse) {
      receivePort.close();
      isolate.kill();
      throw Exception(readyMsg.message);
    }

    final ready = readyMsg as _ReadyResponse;
    return OrtIsolateSession._(
      isolate: isolate,
      workerPort: workerPort,
      responses: broadcastStream,
      inputNames: ready.inputNames,
      outputNames: ready.outputNames,
    );
  }

  List<String> get inputNames => List.unmodifiable(_inputNames);
  List<String> get outputNames => List.unmodifiable(_outputNames);
  int get inputCount => _inputNames.length;
  int get outputCount => _outputNames.length;
  bool get isDisposed => _disposed;

  /// Runs inference in the background isolate.
  ///
  /// Returns output tensors as [Float32List]s. This method is fully async and
  /// safe to call from the UI thread.
  Future<List<Float32List>> runFloat(
    Map<String, OrtIsolateInput> inputs,
    List<int> outputElementCounts,
  ) async {
    _ensureNotDisposed();

    final tensorInputs = inputs.map(
      (name, input) => MapEntry(
        name,
        _TensorData(shape: input.shape, data: input.data),
      ),
    );

    _workerPort.send(_InferRequest(
      inputs: tensorInputs,
      outputElementCounts: outputElementCounts,
    ));

    final response = await _responses.first;
    if (response is _ErrorResponse) {
      throw Exception(response.message);
    }
    return (response as _InferResponse).outputs;
  }

  /// Disposes the session and kills the background isolate.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;

    _workerPort.send(_DisposeRequest());
    await _responses.firstWhere((msg) => msg is _DisposedResponse);

    _isolate.kill();
  }

  void _ensureNotDisposed() {
    if (_disposed) {
      throw StateError('OrtIsolateSession has already been disposed.');
    }
  }
}

// ---------------------------------------------------------------------------
// Internal message types (not exported)
// ---------------------------------------------------------------------------

class _TensorData {
  final List<int> shape;
  final Float32List data;
  _TensorData({required this.shape, required this.data});
}

sealed class _WorkerRequest {}

class _InferRequest extends _WorkerRequest {
  final Map<String, _TensorData> inputs;
  final List<int> outputElementCounts;
  _InferRequest({required this.inputs, required this.outputElementCounts});
}

class _DisposeRequest extends _WorkerRequest {}

sealed class _WorkerResponse {}

class _ReadyResponse extends _WorkerResponse {
  final List<String> inputNames;
  final List<String> outputNames;
  _ReadyResponse({required this.inputNames, required this.outputNames});
}

class _InferResponse extends _WorkerResponse {
  final List<Float32List> outputs;
  _InferResponse({required this.outputs});
}

class _ErrorResponse extends _WorkerResponse {
  final String message;
  _ErrorResponse({required this.message});
}

class _DisposedResponse extends _WorkerResponse {}

// ---------------------------------------------------------------------------
// Worker isolate entry point
// ---------------------------------------------------------------------------

class _WorkerInit {
  final SendPort sendPort;
  final OrtIsolateSessionConfig config;
  _WorkerInit({required this.sendPort, required this.config});
}

void _workerEntryPoint(_WorkerInit init) {
  final mainPort = init.sendPort;
  final config = init.config;
  final receivePort = ReceivePort();

  // Send our SendPort so the main isolate can talk to us.
  mainPort.send(receivePort.sendPort);

  // Initialize runtime inside this isolate.
  OnnxRuntime? runtime;
  OrtSessionWrapper? session;

  try {
    runtime = OnnxRuntime.create();
    runtime.initialize(libraryPath: config.libraryPath);
    runtime.createEnvironment(
      logLevel: config.logLevel,
      logId: config.logId,
    );

    if (config.providers != null) {
      session = OrtSessionWrapper.createWithProviders(
        config.modelPath,
        runtime: runtime,
        providers: config.providers!,
        providerOptions: config.providerOptions,
      );
    } else {
      session = OrtSessionWrapper.create(
        config.modelPath,
        runtime: runtime,
        providerOptions: config.providerOptions,
      );
    }

    mainPort.send(_ReadyResponse(
      inputNames: session.inputNames,
      outputNames: session.outputNames,
    ));
  } catch (e) {
    mainPort.send(_ErrorResponse(message: e.toString()));
    return;
  }

  // Message loop
  receivePort.listen((message) {
    if (message is _InferRequest) {
      try {
        final wrapperInputs = <String, OrtValueWrapper>{};
        for (final entry in message.inputs.entries) {
          wrapperInputs[entry.key] = OrtValueWrapper.fromFloat(
            runtime!,
            entry.value.shape,
            entry.value.data,
          );
        }

        final results = session!.runFloat(
          wrapperInputs,
          message.outputElementCounts,
        );

        for (final wrapper in wrapperInputs.values) {
          wrapper.release();
        }

        mainPort.send(_InferResponse(outputs: results));
      } catch (e) {
        mainPort.send(_ErrorResponse(message: e.toString()));
      }
    } else if (message is _DisposeRequest) {
      session?.dispose();
      runtime?.dispose();
      mainPort.send(_DisposedResponse());
      receivePort.close();
    }
  });
}
