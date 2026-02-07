import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'bindings/onnxruntime_generated.dart';
import 'onnx_runtinme.dart';

class OrtProviders {
  final OnnxRuntime _runtime;

  OrtProviders(this._runtime);

  void appendExecutionProvider(
    Pointer<OrtSessionOptions> options,
    String providerName, {
    Map<String, String> providerOptions = const {},
  }) {
    _runtime.ensureInitialized();
    final api = _runtime.api.ref;

    final providerNameNative = providerName.toNativeUtf8();
    final numKeys = providerOptions.length;

    final keys = providerOptions.keys.toList();
    final values = providerOptions.values.toList();

    final keysPtr = calloc<Pointer<Char>>(numKeys);
    final valuesPtr = calloc<Pointer<Char>>(numKeys);
    final nativeKeys = <Pointer<Utf8>>[];
    final nativeValues = <Pointer<Utf8>>[];

    try {
      for (var i = 0; i < numKeys; i++) {
        final nk = keys[i].toNativeUtf8();
        final nv = values[i].toNativeUtf8();
        nativeKeys.add(nk);
        nativeValues.add(nv);
        keysPtr[i] = nk.cast();
        valuesPtr[i] = nv.cast();
      }

      final status = api.SessionOptionsAppendExecutionProvider
          .asFunction<
            Pointer<OrtStatus> Function(
              Pointer<OrtSessionOptions>,
              Pointer<Char>,
              Pointer<Pointer<Char>>,
              Pointer<Pointer<Char>>,
              int,
            )
          >()(options, providerNameNative.cast(), keysPtr, valuesPtr, numKeys);
      _checkStatus(status);
    } finally {
      calloc.free(providerNameNative);
      for (final nk in nativeKeys) {
        calloc.free(nk);
      }
      for (final nv in nativeValues) {
        calloc.free(nv);
      }
      calloc.free(keysPtr);
      calloc.free(valuesPtr);
    }
  }

  void appendCuda(
    Pointer<OrtSessionOptions> options, {
    int deviceId = 0,
  }) {
    _runtime.ensureInitialized();
    final status = _runtime.bindings
        .OrtSessionOptionsAppendExecutionProvider_CUDA(options, deviceId);
    _checkStatus(status);
  }

  void appendTensorRT(
    Pointer<OrtSessionOptions> options, {
    int deviceId = 0,
  }) {
    _runtime.ensureInitialized();
    final status = _runtime.bindings
        .OrtSessionOptionsAppendExecutionProvider_Tensorrt(options, deviceId);
    _checkStatus(status);
  }

  void appendROCm(
    Pointer<OrtSessionOptions> options, {
    int deviceId = 0,
  }) {
    _runtime.ensureInitialized();
    final status = _runtime.bindings
        .OrtSessionOptionsAppendExecutionProvider_ROCM(options, deviceId);
    _checkStatus(status);
  }

  void appendCoreML(
    Pointer<OrtSessionOptions> options, {
    Map<String, String> providerOptions = const {},
  }) {
    appendExecutionProvider(options, 'CoreML', providerOptions: providerOptions);
  }

  void appendOpenVINO(
    Pointer<OrtSessionOptions> options, {
    Map<String, String> providerOptions = const {},
  }) {
    appendExecutionProvider(
      options,
      'OpenVINO',
      providerOptions: providerOptions,
    );
  }

  void appendDnnl(
    Pointer<OrtSessionOptions> options, {
    int useArena = 1,
  }) {
    _runtime.ensureInitialized();
    final status = _runtime.bindings
        .OrtSessionOptionsAppendExecutionProvider_Dnnl(options, useArena);
    _checkStatus(status);
  }

  void appendQnn(
    Pointer<OrtSessionOptions> options, {
    Map<String, String> providerOptions = const {},
  }) {
    appendExecutionProvider(options, 'QNN', providerOptions: providerOptions);
  }

  void appendXnnpack(
    Pointer<OrtSessionOptions> options, {
    Map<String, String> providerOptions = const {},
  }) {
    appendExecutionProvider(
      options,
      'XNNPACK',
      providerOptions: providerOptions,
    );
  }

  void _checkStatus(Pointer<OrtStatus> status) {
    if (status == nullptr) return;

    final api = _runtime.api.ref;
    final errorMsg = api.GetErrorMessage
        .asFunction<Pointer<Char> Function(Pointer<OrtStatus>)>()(status);
    final errorString = errorMsg.cast<Utf8>().toDartString();

    api.ReleaseStatus
        .asFunction<void Function(Pointer<OrtStatus>)>()(status);

    throw Exception('OrtProviders Error: $errorString');
  }
}
