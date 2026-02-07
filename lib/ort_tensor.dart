import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings/onnxruntime_generated.dart';
import 'onnx_runtinme.dart';

class OrtTensor {
  final OnnxRuntime _runtime;

  OrtTensor(this._runtime);

  Pointer<OrtMemoryInfo> createCpuMemoryInfo() {
    _runtime.ensureInitialized();
    final api = _runtime.api.ref;
    final memInfoPtr = calloc<Pointer<OrtMemoryInfo>>();
    try {
      final status = api.CreateCpuMemoryInfo
          .asFunction<
            Pointer<OrtStatus> Function(
              int type,
              int memType,
              Pointer<Pointer<OrtMemoryInfo>>,
            )
          >()(
            OrtAllocatorType.OrtArenaAllocator.value,
            OrtMemType.OrtMemTypeDefault.value,
            memInfoPtr,
          );
      _checkStatus(status);
      return memInfoPtr.value;
    } finally {
      calloc.free(memInfoPtr);
    }
  }

  void releaseMemoryInfo(Pointer<OrtMemoryInfo> memInfo) {
    if (memInfo == nullptr) return;
    _runtime.api.ref.ReleaseMemoryInfo
        .asFunction<void Function(Pointer<OrtMemoryInfo>)>()(memInfo);
  }

  Pointer<OrtValue> createFloat(List<int> shape, Float32List data) {
    _runtime.ensureInitialized();
    final api = _runtime.api.ref;
    final memInfo = createCpuMemoryInfo();
    final valuePtr = calloc<Pointer<OrtValue>>();
    final shapePtr = calloc<Int64>(shape.length);
    final dataPtr = calloc<Float>(data.length);

    try {
      for (var i = 0; i < shape.length; i++) {
        shapePtr[i] = shape[i];
      }
      for (var i = 0; i < data.length; i++) {
        dataPtr[i] = data[i];
      }

      final status = api.CreateTensorWithDataAsOrtValue
          .asFunction<
            Pointer<OrtStatus> Function(
              Pointer<OrtMemoryInfo>,
              Pointer<Void>,
              int,
              Pointer<Int64>,
              int,
              int,
              Pointer<Pointer<OrtValue>>,
            )
          >()(
            memInfo,
            dataPtr.cast(),
            data.length * sizeOf<Float>(),
            shapePtr,
            shape.length,
            ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT.value,
            valuePtr,
          );
      _checkStatus(status);
      return valuePtr.value;
    } finally {
      releaseMemoryInfo(memInfo);
      calloc.free(valuePtr);
      calloc.free(shapePtr);
    }
  }

  Pointer<OrtValue> createInt64(List<int> shape, List<int> data) {
    _runtime.ensureInitialized();
    final api = _runtime.api.ref;
    final memInfo = createCpuMemoryInfo();
    final valuePtr = calloc<Pointer<OrtValue>>();
    final shapePtr = calloc<Int64>(shape.length);
    final dataPtr = calloc<Int64>(data.length);

    try {
      for (var i = 0; i < shape.length; i++) {
        shapePtr[i] = shape[i];
      }
      for (var i = 0; i < data.length; i++) {
        dataPtr[i] = data[i];
      }

      final status = api.CreateTensorWithDataAsOrtValue
          .asFunction<
            Pointer<OrtStatus> Function(
              Pointer<OrtMemoryInfo>,
              Pointer<Void>,
              int,
              Pointer<Int64>,
              int,
              int,
              Pointer<Pointer<OrtValue>>,
            )
          >()(
            memInfo,
            dataPtr.cast(),
            data.length * sizeOf<Int64>(),
            shapePtr,
            shape.length,
            ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64.value,
            valuePtr,
          );
      _checkStatus(status);
      return valuePtr.value;
    } finally {
      releaseMemoryInfo(memInfo);
      calloc.free(valuePtr);
      calloc.free(shapePtr);
    }
  }

  Float32List getDataFloat(Pointer<OrtValue> value, int elementCount) {
    _runtime.ensureInitialized();
    final dataPtr = calloc<Pointer<Void>>();
    try {
      final status = _runtime.api.ref.GetTensorMutableData
          .asFunction<
            Pointer<OrtStatus> Function(
              Pointer<OrtValue>,
              Pointer<Pointer<Void>>,
            )
          >()(value, dataPtr);
      _checkStatus(status);
      return dataPtr.value.cast<Float>().asTypedList(elementCount);
    } finally {
      calloc.free(dataPtr);
    }
  }

  void release(Pointer<OrtValue> value) {
    if (value == nullptr) return;
    _runtime.api.ref.ReleaseValue
        .asFunction<void Function(Pointer<OrtValue>)>()(value);
  }

  void _checkStatus(Pointer<OrtStatus> status) {
    if (status == nullptr) return;

    final api = _runtime.api.ref;
    final errorMsg = api.GetErrorMessage
        .asFunction<Pointer<Char> Function(Pointer<OrtStatus>)>()(status);
    final errorString = errorMsg.cast<Utf8>().toDartString();

    api.ReleaseStatus
        .asFunction<void Function(Pointer<OrtStatus>)>()(status);

    throw Exception('OrtTensor Error: $errorString');
  }
}
