import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import 'bindings/onnxruntime_generated.dart';
import 'onnx_runtinme.dart';

enum OrtProvider {
  cpu('CPUExecutionProvider'),
  coreML('CoreMLExecutionProvider'),
  cuda('CUDAExecutionProvider'),
  tensorRT('TensorrtExecutionProvider'),
  rocm('ROCMExecutionProvider'),
  openVINO('OpenVINOExecutionProvider'),
  dnnl('DnnlExecutionProvider'),
  qnn('QNNExecutionProvider'),
  xnnpack('XNNPACKExecutionProvider');

  final String ortName;
  const OrtProvider(this.ortName);
}

class OrtProviders {
  final OnnxRuntime _runtime;

  OrtProviders(this._runtime);

  List<String> getAvailableProviders() {
    _runtime.ensureInitialized();
    final api = _runtime.api.ref;

    final outPtr = calloc<Pointer<Pointer<Char>>>();
    final lengthPtr = calloc<Int>();

    try {
      final status = api.GetAvailableProviders
          .asFunction<
            Pointer<OrtStatus> Function(
              Pointer<Pointer<Pointer<Char>>>,
              Pointer<Int>,
            )
          >()(outPtr, lengthPtr);
      _checkStatus(status);

      final length = lengthPtr.value;
      final providers = <String>[];
      final arrayPtr = outPtr.value;

      for (var i = 0; i < length; i++) {
        providers.add(arrayPtr[i].cast<Utf8>().toDartString());
      }

      api.ReleaseAvailableProviders
          .asFunction<
            Pointer<OrtStatus> Function(Pointer<Pointer<Char>>, int)
          >()(arrayPtr, length);

      return providers;
    } finally {
      calloc.free(outPtr);
      calloc.free(lengthPtr);
    }
  }

  bool isProviderAvailable(OrtProvider provider) {
    return getAvailableProviders().contains(provider.ortName);
  }

  List<OrtProvider> getDefaultProvidersForPlatform() {
    final available = getAvailableProviders();

    if (Platform.isIOS || Platform.isMacOS) {
      final providers = <OrtProvider>[];
      if (available.contains(OrtProvider.coreML.ortName)) {
        providers.add(OrtProvider.coreML);
      }
      providers.add(OrtProvider.cpu);
      return providers;
    }

    if (Platform.isAndroid) {
      final providers = <OrtProvider>[];
      if (available.contains(OrtProvider.xnnpack.ortName)) {
        providers.add(OrtProvider.xnnpack);
      }
      if (available.contains(OrtProvider.qnn.ortName)) {
        providers.add(OrtProvider.qnn);
      }
      providers.add(OrtProvider.cpu);
      return providers;
    }

    if (Platform.isLinux || Platform.isWindows) {
      final providers = <OrtProvider>[];
      if (available.contains(OrtProvider.cuda.ortName)) {
        providers.add(OrtProvider.cuda);
      }
      if (available.contains(OrtProvider.tensorRT.ortName)) {
        providers.add(OrtProvider.tensorRT);
      }
      if (available.contains(OrtProvider.rocm.ortName)) {
        providers.add(OrtProvider.rocm);
      }
      if (available.contains(OrtProvider.openVINO.ortName)) {
        providers.add(OrtProvider.openVINO);
      }
      if (available.contains(OrtProvider.dnnl.ortName)) {
        providers.add(OrtProvider.dnnl);
      }
      providers.add(OrtProvider.cpu);
      return providers;
    }

    return [OrtProvider.cpu];
  }

  void appendProvider(
    Pointer<OrtSessionOptions> options,
    OrtProvider provider, {
    Map<String, String> providerOptions = const {},
  }) {
    switch (provider) {
      case OrtProvider.cpu:
        break;
      case OrtProvider.coreML:
        _appendViaGenericApi(options, 'CoreML',
            providerOptions: providerOptions);
      case OrtProvider.cuda:
        _appendViaGenericApi(options, 'CUDA',
            providerOptions: providerOptions);
      case OrtProvider.tensorRT:
        _appendViaGenericApi(options, 'TensorRT',
            providerOptions: providerOptions);
      case OrtProvider.rocm:
        _appendViaGenericApi(options, 'ROCM',
            providerOptions: providerOptions);
      case OrtProvider.openVINO:
        _appendViaGenericApi(options, 'OpenVINO',
            providerOptions: providerOptions);
      case OrtProvider.dnnl:
        _appendViaGenericApi(options, 'DNNL',
            providerOptions: providerOptions);
      case OrtProvider.qnn:
        _appendViaGenericApi(options, 'QNN',
            providerOptions: providerOptions);
      case OrtProvider.xnnpack:
        _appendViaGenericApi(options, 'XNNPACK',
            providerOptions: providerOptions);
    }
  }

  void appendProviders(
    Pointer<OrtSessionOptions> options,
    List<OrtProvider> providers, {
    Map<OrtProvider, Map<String, String>> providerOptions = const {},
  }) {
    for (final provider in providers) {
      appendProvider(
        options,
        provider,
        providerOptions: providerOptions[provider] ?? const {},
      );
    }
  }

  void appendDefaultProviders(
    Pointer<OrtSessionOptions> options, {
    Map<OrtProvider, Map<String, String>> providerOptions = const {},
  }) {
    appendProviders(
      options,
      getDefaultProvidersForPlatform(),
      providerOptions: providerOptions,
    );
  }

  bool tryAppendProvider(
    Pointer<OrtSessionOptions> options,
    OrtProvider provider, {
    Map<String, String> providerOptions = const {},
  }) {
    if (!isProviderAvailable(provider)) return false;
    try {
      appendProvider(options, provider, providerOptions: providerOptions);
      return true;
    } catch (_) {
      return false;
    }
  }

  void appendExecutionProvider(
    Pointer<OrtSessionOptions> options,
    String providerName, {
    Map<String, String> providerOptions = const {},
  }) {
    _appendViaGenericApi(options, providerName,
        providerOptions: providerOptions);
  }

  void _appendViaGenericApi(
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
