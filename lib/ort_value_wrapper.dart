import 'dart:ffi';
import 'dart:typed_data';

import 'bindings/onnxruntime_generated.dart';
import 'onnx_runtinme.dart';
import 'ort_tensor.dart';

class OrtValueWrapper {
  final Pointer<OrtValue> _ptr;
  final OrtTensor _tensor;
  bool _released = false;

  OrtValueWrapper._(this._ptr, this._tensor);

  static OrtValueWrapper fromFloat(
    OnnxRuntime runtime,
    List<int> shape,
    Float32List data,
  ) {
    final tensor = OrtTensor(runtime);
    final ptr = tensor.createFloat(shape, data);
    return OrtValueWrapper._(ptr, tensor);
  }

  static OrtValueWrapper fromInt64(
    OnnxRuntime runtime,
    List<int> shape,
    List<int> data,
  ) {
    final tensor = OrtTensor(runtime);
    final ptr = tensor.createInt64(shape, data);
    return OrtValueWrapper._(ptr, tensor);
  }

  static OrtValueWrapper fromPointer(OnnxRuntime runtime, Pointer<OrtValue> ptr) {
    return OrtValueWrapper._(ptr, OrtTensor(runtime));
  }

  Float32List toFloatList(int elementCount) {
    _ensureNotReleased();
    return _tensor.getDataFloat(_ptr, elementCount);
  }

  Pointer<OrtValue> get pointer {
    _ensureNotReleased();
    return _ptr;
  }

  void release() {
    if (_released) return;
    _tensor.release(_ptr);
    _released = true;
  }

  bool get isReleased => _released;

  void _ensureNotReleased() {
    if (_released) {
      throw StateError('OrtValueWrapper has already been released.');
    }
  }
}
