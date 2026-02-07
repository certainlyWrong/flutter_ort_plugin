import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import 'bindings/onnxruntime_generated.dart';

class OnnxRuntime {
  late OnnxRuntimeBindings _bindings;
  late Pointer<OrtApi> _api;
  Pointer<OrtEnv>? _env;
  bool _initialized = false;

  OnnxRuntime._();

  static OnnxRuntime? _instance;

  static OnnxRuntime get instance {
    _instance ??= OnnxRuntime._();
    return _instance!;
  }

  void initialize({String? libraryPath}) {
    if (_initialized) {
      return;
    }

    final DynamicLibrary dylib;
    if (libraryPath != null) {
      dylib = DynamicLibrary.open(libraryPath);
    } else {
      if (Platform.isAndroid) {
        dylib = DynamicLibrary.open('libonnxruntime.so');
      } else if (Platform.isIOS || Platform.isMacOS) {
        dylib = DynamicLibrary.process();
      } else if (Platform.isLinux) {
        dylib = DynamicLibrary.open('libonnxruntime.so');
      } else if (Platform.isWindows) {
        dylib = DynamicLibrary.open('onnxruntime.dll');
      } else {
        throw UnsupportedError('Platform not supported');
      }
    }

    _bindings = OnnxRuntimeBindings(dylib);

    final apiBase = _bindings.OrtGetApiBase();
    if (apiBase == nullptr) {
      throw Exception('Failed to get OrtApiBase');
    }

    final getApi = apiBase.ref.GetApi;
    _api = getApi.asFunction<Pointer<OrtApi> Function(int)>()(ORT_API_VERSION);

    if (_api == nullptr) {
      throw Exception('Failed to get OrtApi for version $ORT_API_VERSION');
    }

    _initialized = true;
  }

  void createEnvironment({int logLevel = 3, String logId = 'OnnxRuntime'}) {
    if (!_initialized) {
      throw StateError('OnnxRuntime not initialized. Call initialize() first.');
    }

    if (_env != nullptr && _env != null) {
      return;
    }

    final envPtr = calloc<Pointer<OrtEnv>>();
    final logIdNative = logId.toNativeUtf8();

    try {
      final createEnv = _api.ref.CreateEnv;
      final status = createEnv
          .asFunction<
            Pointer<OrtStatus> Function(
              int logLevel,
              Pointer<Char> logId,
              Pointer<Pointer<OrtEnv>> out,
            )
          >()(logLevel, logIdNative.cast(), envPtr);

      _checkStatus(status);
      _env = envPtr.value;
    } finally {
      calloc.free(logIdNative);
      calloc.free(envPtr);
    }
  }

  Pointer<OrtSessionOptions> createSessionOptions() {
    if (!_initialized) {
      throw StateError('OnnxRuntime not initialized. Call initialize() first.');
    }

    final optionsPtr = calloc<Pointer<OrtSessionOptions>>();

    try {
      final createOptions = _api.ref.CreateSessionOptions;
      final status = createOptions
          .asFunction<
            Pointer<OrtStatus> Function(Pointer<Pointer<OrtSessionOptions>> out)
          >()(optionsPtr);

      _checkStatus(status);
      return optionsPtr.value;
    } finally {
      calloc.free(optionsPtr);
    }
  }

  Pointer<OrtSession> createSession(
    String modelPath,
    Pointer<OrtSessionOptions> sessionOptions,
  ) {
    if (!_initialized) {
      throw StateError('OnnxRuntime not initialized. Call initialize() first.');
    }

    if (_env == null || _env == nullptr) {
      throw StateError(
        'Environment not created. Call createEnvironment() first.',
      );
    }

    final sessionPtr = calloc<Pointer<OrtSession>>();
    final modelPathNative = modelPath.toNativeUtf8();

    try {
      final createSession = _api.ref.CreateSession;
      final status = createSession
          .asFunction<
            Pointer<OrtStatus> Function(
              Pointer<OrtEnv> env,
              Pointer<Char> modelPath,
              Pointer<OrtSessionOptions> options,
              Pointer<Pointer<OrtSession>> out,
            )
          >()(_env!, modelPathNative.cast(), sessionOptions, sessionPtr);

      _checkStatus(status);
      return sessionPtr.value;
    } finally {
      calloc.free(modelPathNative);
      calloc.free(sessionPtr);
    }
  }

  void releaseSession(Pointer<OrtSession> session) {
    if (!_initialized || session == nullptr) {
      return;
    }

    final releaseSession = _api.ref.ReleaseSession;
    releaseSession.asFunction<void Function(Pointer<OrtSession>)>()(session);
  }

  void releaseSessionOptions(Pointer<OrtSessionOptions> options) {
    if (!_initialized || options == nullptr) {
      return;
    }

    final releaseOptions = _api.ref.ReleaseSessionOptions;
    releaseOptions.asFunction<void Function(Pointer<OrtSessionOptions>)>()(
      options,
    );
  }

  void ensureInitialized() {
    if (!_initialized) {
      throw StateError('OnnxRuntime not initialized. Call initialize() first.');
    }
  }

  Pointer<OrtAllocator> getDefaultAllocator() {
    ensureInitialized();
    final allocatorPtr = calloc<Pointer<OrtAllocator>>();
    try {
      final status = _api.ref.GetAllocatorWithDefaultOptions
          .asFunction<
            Pointer<OrtStatus> Function(Pointer<Pointer<OrtAllocator>>)
          >()(allocatorPtr);
      _checkStatus(status);
      return allocatorPtr.value;
    } finally {
      calloc.free(allocatorPtr);
    }
  }

  int getSessionInputCount(Pointer<OrtSession> session) {
    ensureInitialized();
    final countPtr = calloc<Size>();
    try {
      final status = _api.ref.SessionGetInputCount
          .asFunction<
            Pointer<OrtStatus> Function(Pointer<OrtSession>, Pointer<Size>)
          >()(session, countPtr);
      _checkStatus(status);
      return countPtr.value;
    } finally {
      calloc.free(countPtr);
    }
  }

  int getSessionOutputCount(Pointer<OrtSession> session) {
    ensureInitialized();
    final countPtr = calloc<Size>();
    try {
      final status = _api.ref.SessionGetOutputCount
          .asFunction<
            Pointer<OrtStatus> Function(Pointer<OrtSession>, Pointer<Size>)
          >()(session, countPtr);
      _checkStatus(status);
      return countPtr.value;
    } finally {
      calloc.free(countPtr);
    }
  }

  String getSessionInputName(Pointer<OrtSession> session, int index) {
    ensureInitialized();
    final allocator = getDefaultAllocator();
    final namePtr = calloc<Pointer<Char>>();
    try {
      final status = _api.ref.SessionGetInputName
          .asFunction<
            Pointer<OrtStatus> Function(
              Pointer<OrtSession>,
              int,
              Pointer<OrtAllocator>,
              Pointer<Pointer<Char>>,
            )
          >()(session, index, allocator, namePtr);
      _checkStatus(status);
      final name = namePtr.value.cast<Utf8>().toDartString();
      _api.ref.AllocatorFree
          .asFunction<
            Pointer<OrtStatus> Function(Pointer<OrtAllocator>, Pointer<Void>)
          >()(allocator, namePtr.value.cast());
      return name;
    } finally {
      calloc.free(namePtr);
    }
  }

  String getSessionOutputName(Pointer<OrtSession> session, int index) {
    ensureInitialized();
    final allocator = getDefaultAllocator();
    final namePtr = calloc<Pointer<Char>>();
    try {
      final status = _api.ref.SessionGetOutputName
          .asFunction<
            Pointer<OrtStatus> Function(
              Pointer<OrtSession>,
              int,
              Pointer<OrtAllocator>,
              Pointer<Pointer<Char>>,
            )
          >()(session, index, allocator, namePtr);
      _checkStatus(status);
      final name = namePtr.value.cast<Utf8>().toDartString();
      _api.ref.AllocatorFree
          .asFunction<
            Pointer<OrtStatus> Function(Pointer<OrtAllocator>, Pointer<Void>)
          >()(allocator, namePtr.value.cast());
      return name;
    } finally {
      calloc.free(namePtr);
    }
  }

  List<String> getSessionInputNames(Pointer<OrtSession> session) {
    final count = getSessionInputCount(session);
    return List.generate(count, (i) => getSessionInputName(session, i));
  }

  List<String> getSessionOutputNames(Pointer<OrtSession> session) {
    final count = getSessionOutputCount(session);
    return List.generate(count, (i) => getSessionOutputName(session, i));
  }

  Pointer<OrtRunOptions> createRunOptions() {
    ensureInitialized();
    final runOptionsPtr = calloc<Pointer<OrtRunOptions>>();
    try {
      final status = _api.ref.CreateRunOptions
          .asFunction<
            Pointer<OrtStatus> Function(Pointer<Pointer<OrtRunOptions>>)
          >()(runOptionsPtr);
      _checkStatus(status);
      return runOptionsPtr.value;
    } finally {
      calloc.free(runOptionsPtr);
    }
  }

  void releaseRunOptions(Pointer<OrtRunOptions> runOptions) {
    if (!_initialized || runOptions == nullptr) return;
    _api.ref.ReleaseRunOptions
        .asFunction<void Function(Pointer<OrtRunOptions>)>()(runOptions);
  }

  List<Pointer<OrtValue>> run(
    Pointer<OrtSession> session, {
    required List<String> inputNames,
    required List<Pointer<OrtValue>> inputValues,
    required List<String> outputNames,
    Pointer<OrtRunOptions>? runOptions,
  }) {
    ensureInitialized();

    final createdRunOptions = runOptions == null;
    final actualRunOptions = runOptions ?? createRunOptions();

    final inputNamesPtr = calloc<Pointer<Char>>(inputNames.length);
    final inputValuesPtr = calloc<Pointer<OrtValue>>(inputValues.length);
    final outputNamesPtr = calloc<Pointer<Char>>(outputNames.length);
    final outputValuesPtr = calloc<Pointer<OrtValue>>(outputNames.length);

    final nativeInputNames = <Pointer<Utf8>>[];
    final nativeOutputNames = <Pointer<Utf8>>[];

    try {
      for (var i = 0; i < inputNames.length; i++) {
        final nativeName = inputNames[i].toNativeUtf8();
        nativeInputNames.add(nativeName);
        inputNamesPtr[i] = nativeName.cast();
      }
      for (var i = 0; i < inputValues.length; i++) {
        inputValuesPtr[i] = inputValues[i];
      }
      for (var i = 0; i < outputNames.length; i++) {
        final nativeName = outputNames[i].toNativeUtf8();
        nativeOutputNames.add(nativeName);
        outputNamesPtr[i] = nativeName.cast();
      }
      for (var i = 0; i < outputNames.length; i++) {
        outputValuesPtr[i] = nullptr;
      }

      final status =
          _api.ref.Run
              .asFunction<
                Pointer<OrtStatus> Function(
                  Pointer<OrtSession>,
                  Pointer<OrtRunOptions>,
                  Pointer<Pointer<Char>>,
                  Pointer<Pointer<OrtValue>>,
                  int,
                  Pointer<Pointer<Char>>,
                  int,
                  Pointer<Pointer<OrtValue>>,
                )
              >()(
            session,
            actualRunOptions,
            inputNamesPtr,
            inputValuesPtr,
            inputNames.length,
            outputNamesPtr,
            outputNames.length,
            outputValuesPtr,
          );
      _checkStatus(status);

      final results = <Pointer<OrtValue>>[];
      for (var i = 0; i < outputNames.length; i++) {
        results.add(outputValuesPtr[i]);
      }
      return results;
    } finally {
      for (final name in nativeInputNames) {
        calloc.free(name);
      }
      for (final name in nativeOutputNames) {
        calloc.free(name);
      }
      calloc.free(inputNamesPtr);
      calloc.free(inputValuesPtr);
      calloc.free(outputNamesPtr);
      calloc.free(outputValuesPtr);
      if (createdRunOptions) {
        releaseRunOptions(actualRunOptions);
      }
    }
  }

  void _checkStatus(Pointer<OrtStatus> status) {
    if (status == nullptr) {
      return;
    }

    final getErrorMessage = _api.ref.GetErrorMessage;
    final errorMsg = getErrorMessage
        .asFunction<Pointer<Char> Function(Pointer<OrtStatus>)>()(status);
    final errorString = errorMsg.cast<Utf8>().toDartString();

    final releaseStatus = _api.ref.ReleaseStatus;
    releaseStatus.asFunction<void Function(Pointer<OrtStatus>)>()(status);

    throw Exception('OnnxRuntime Error: $errorString');
  }

  void dispose() {
    if (!_initialized) {
      return;
    }

    if (_env != null && _env != nullptr) {
      final releaseEnv = _api.ref.ReleaseEnv;
      releaseEnv.asFunction<void Function(Pointer<OrtEnv>)>()(_env!);
      _env = null;
    }

    _initialized = false;
  }

  bool get isInitialized => _initialized;
  Pointer<OrtApi> get api => _api;
  OnnxRuntimeBindings get bindings => _bindings;
  Pointer<OrtEnv>? get environment => _env;
}
