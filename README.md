# Flutter ONNX Runtime Plugin

[![Flutter Version](https://img.shields.io/badge/Flutter-%5E3.3.0-blue)](https://flutter.dev)
[![Dart Version](https://img.shields.io/badge/Dart-%5E3.10.8-blue)](https://dart.dev)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.24.1-green)](https://onnxruntime.ai)

A Flutter FFI (Foreign Function Interface) plugin for running ONNX (Open Neural Network Exchange) models directly in Flutter applications. This plugin uses the native ONNX Runtime C API library for machine learning inference on multiple platforms.

## 🚀 Features

- **ONNX Model Execution**: Load and execute ONNX models (.onnx) directly in Flutter
- **Multiplatform Support**: Android, iOS, macOS, Linux, and Windows
- **Native FFI API**: Direct integration with the ONNX Runtime C API via Dart FFI
- **High-Level APIs**: User-friendly `OrtSessionWrapper` and `OrtValueWrapper` classes without FFI pointers
- **Execution Providers**: Support for CUDA, TensorRT, ROCm, CoreML, OpenVINO, QNN, XNNPACK, and oneDNN
- **Auto-generated Bindings**: Uses `ffigen` to generate Dart bindings from C headers
- **Resource Management**: Automatic lifecycle management for sessions, tensors, and environments
- **Typed Tensor Creation**: Easy creation of Float32 and Int64 tensors with `OrtTensor` and `OrtValueWrapper`

## 📋 Requirements

- **Flutter**: >= 3.3.0
- **Dart**: >= 3.10.8
- **CMake**: >= 3.10 (Linux/Windows)
- **Android NDK**: (for Android builds)
- **Xcode**: (for iOS/macOS builds)

## 📦 Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_ort_plugin:
    git:
      url: https://github.com/adrinator/flutter_ort_plugin.git
```

Or run:

```bash
flutter pub add flutter_ort_plugin
```

## 🎯 Basic Usage

### Initialization

```dart
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

// Get singleton instance
final runtime = OnnxRuntime.instance;

// Initialize the runtime (loads the native library)
runtime.initialize();

// Create ONNX Runtime environment
runtime.createEnvironment(
  logLevel: 3,  // Log level (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL)
  logId: 'MyApp',
);
```

### High-Level API (Recommended)

The plugin provides high-level wrapper classes that hide FFI pointers and manage resources automatically:

#### OrtSessionWrapper - Complete Model Loading & Inference

```dart
// Load a model with optional provider configuration
final session = OrtSessionWrapper.create(
  'path/to/model.onnx',
  configureProviders: (providers, options) {
    // Optional: Add execution providers
    // providers.appendCuda(options, deviceId: 0);
    // providers.appendCoreML(options);
  },
);

// Get model metadata
print('Inputs: ${session.inputNames}');  // Auto-discovered
print('Outputs: ${session.outputNames}');

// Create input tensors (no FFI pointers!)
final inputData = Float32List.fromList([1.0, 2.0, 3.0]);
final inputTensor = OrtValueWrapper.fromFloat(
  runtime,
  [1, 3],  // shape: [batch=1, features=3]
  inputData,
);

// Run inference and get outputs as Dart lists
final outputs = session.runFloat(
  {'input_name': inputTensor},
  [10],  // output element counts
);

// outputs is List<Float32List> - pure Dart types!
final predictions = outputs.first;
print('Predicted class: ${predictions.indexOf(predictions.reduce(max))}');

// Cleanup
inputTensor.release();
session.dispose();
```

#### Execution Providers (GPU Acceleration)

```dart
final providers = OrtProviders(runtime);

// CUDA (NVIDIA GPU)
providers.appendCuda(options, deviceId: 0);

// CoreML (Apple Neural Engine / GPU)
providers.appendCoreML(options, providerOptions: {
  'MLComputeUnits': 'All',
});

// OpenVINO (Intel hardware)
providers.appendOpenVINO(options, providerOptions: {
  'device_type': 'CPU',
});

// ROCm (AMD GPU)
providers.appendROCm(options, deviceId: 0);

// TensorRT (NVIDIA optimized)
providers.appendTensorRT(options, deviceId: 0);

// Generic provider (any registered provider)
providers.appendExecutionProvider(
  options,
  'QNN',  // Qualcomm AI Engine
  providerOptions: {'backend_path': '/path/to/libQnnCpu.so'},
);
```

### Low-Level API (Direct FFI)

For advanced use cases requiring direct FFI control:

```dart
// Create session options
final sessionOptions = runtime.createSessionOptions();

// Load ONNX model
final session = runtime.createSession(
  'assets/model.onnx',
  sessionOptions,
);

// Get input/output names
final inputNames = runtime.getSessionInputNames(session);
final outputNames = runtime.getSessionOutputNames(session);

// Create tensors using OrtTensor helper
final tensor = OrtTensor(runtime);
final inputValue = tensor.createFloat([1, 3], inputData);

// Run inference
final outputs = runtime.run(
  session,
  inputNames: inputNames,
  inputValues: [inputValue],
  outputNames: outputNames,
);

// Read output data
final outputData = tensor.getDataFloat(outputs.first, 10);

// Cleanup
for (final output in outputs) {
  tensor.release(output);
}
tensor.release(inputValue);
for (final opt in outputs) {
  runtime.releaseValue(opt);  // alternative
}
runtime.releaseSession(session);
runtime.releaseSessionOptions(sessionOptions);
```

### Resource Cleanup

```dart
// Release resources in the correct order
runtime.releaseSession(session);
runtime.releaseSessionOptions(sessionOptions);
runtime.dispose();
```

## 🏗️ Architecture

### Project Structure

```
flutter_ort_plugin/
├── lib/
│   ├── onnx_runtinme.dart          # Core runtime API (OnnxRuntime)
│   ├── ort_tensor.dart             # Low-level tensor operations (OrtTensor)
│   ├── ort_providers.dart          # Execution provider configuration (OrtProviders)
│   ├── ort_session_wrapper.dart    # High-level session wrapper (OrtSessionWrapper)
│   ├── ort_value_wrapper.dart      # High-level tensor wrapper (OrtValueWrapper)
│   ├── flutter_ort_plugin.dart     # Plugin exports
│   └── bindings/
│       └── onnxruntime_generated.dart  # Generated FFI bindings
├── src/                            # Native C/C++ code (if needed)
├── linux/                          # Linux CMake configuration
├── android/                        # Android Gradle configuration
├── ios/                            # iOS CocoaPods configuration
├── macos/                          # macOS configuration
├── windows/                        # Windows configuration
└── example/                        # Example MNIST application
```

### Data Flow

```
[Flutter App]
    ↓ (Dart FFI)
[Generated Bindings]
    ↓ (C API)
[ONNX Runtime C Library]
    ↓
[ONNX Model]
```

## 🔧 Platform Configuration

### Linux

The plugin uses FetchContent to automatically download and link the ONNX Runtime library:

```cmake
# Configuration in linux/CMakeLists.txt
FetchContent_Declare(
  onnxruntime_linux
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-1.24.1.tgz
)
```

### Android

Configure the NDK in `android/build.gradle`:

```gradle
android {
    ndkVersion "25.1.8937393"
    // ...
}
```

### iOS/macOS

Libraries are linked via CocoaPods in the `.podspec` files.

## 📝 API Reference

### High-Level APIs (Recommended)

#### `OrtSessionWrapper` Class

A high-level wrapper for ONNX sessions that handles resource management automatically.

| Method                                                                                               | Description                                                         |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `OrtSessionWrapper.create(String modelPath, {OnnxRuntime? runtime, Function? configureProviders})`   | Factory constructor that loads a model and discovers inputs/outputs |
| `run(Map<String, OrtValueWrapper> inputs)` → `List<OrtValueWrapper>`                                 | Run inference with wrapped tensors                                  |
| `runFloat(Map<String, OrtValueWrapper> inputs, List<int> outputElementCounts)` → `List<Float32List>` | Run inference and get outputs as Dart lists directly                |
| `dispose()`                                                                                          | Release session resources                                           |
| `inputNames` → `List<String>`                                                                        | Auto-discovered input tensor names                                  |
| `outputNames` → `List<String>`                                                                       | Auto-discovered output tensor names                                 |
| `inputCount` → `int`                                                                                 | Number of input tensors                                             |
| `outputCount` → `int`                                                                                | Number of output tensors                                            |

#### `OrtValueWrapper` Class

A high-level wrapper for ONNX tensor values that hides FFI pointers.

| Method                                                                      | Description                                 |
| --------------------------------------------------------------------------- | ------------------------------------------- |
| `OrtValueWrapper.fromFloat(OnnxRuntime, List<int> shape, Float32List data)` | Create a float32 tensor from shape and data |
| `OrtValueWrapper.fromInt64(OnnxRuntime, List<int> shape, List<int> data)`   | Create an int64 tensor from shape and data  |
| `toFloatList(int elementCount)` → `Float32List`                             | Read tensor data as Dart Float32List        |
| `release()`                                                                 | Release native tensor memory                |
| `isReleased` → `bool`                                                       | Check if already released                   |

#### `OrtProviders` Class

Configure execution providers for GPU/hardware acceleration.

| Method                                                                                    | Description                            |
| ----------------------------------------------------------------------------------------- | -------------------------------------- |
| `appendCuda(Pointer<OrtSessionOptions>, {int deviceId = 0})`                              | Add NVIDIA CUDA provider               |
| `appendTensorRT(Pointer<OrtSessionOptions>, {int deviceId = 0})`                          | Add NVIDIA TensorRT provider           |
| `appendROCm(Pointer<OrtSessionOptions>, {int deviceId = 0})`                              | Add AMD ROCm provider                  |
| `appendCoreML(Pointer<OrtSessionOptions>, {Map providerOptions})`                         | Add Apple CoreML provider              |
| `appendOpenVINO(Pointer<OrtSessionOptions>, {Map providerOptions})`                       | Add Intel OpenVINO provider            |
| `appendQnn(Pointer<OrtSessionOptions>, {Map providerOptions})`                            | Add Qualcomm QNN provider              |
| `appendXnnpack(Pointer<OrtSessionOptions>, {Map providerOptions})`                        | Add XNNPACK provider                   |
| `appendDnnl(Pointer<OrtSessionOptions>, {int useArena = 1})`                              | Add oneDNN provider                    |
| `appendExecutionProvider(Pointer<OrtSessionOptions>, String name, {Map providerOptions})` | Generic provider (any registered name) |

### Low-Level APIs

#### `OnnxRuntime` Class

Core runtime class for direct FFI access.

| Method                                                                                        | Description                                           |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `initialize({String? libraryPath})`                                                           | Initializes the runtime by loading the native library |
| `createEnvironment({int logLevel, String logId})`                                             | Creates the ONNX Runtime environment                  |
| `createSessionOptions()` → `Pointer<OrtSessionOptions>`                                       | Creates configuration options for sessions            |
| `createSession(String modelPath, Pointer<OrtSessionOptions> options)` → `Pointer<OrtSession>` | Creates a session from a model                        |
| `getSessionInputNames(Pointer<OrtSession>)` → `List<String>`                                  | Get input tensor names                                |
| `getSessionOutputNames(Pointer<OrtSession>)` → `List<String>`                                 | Get output tensor names                               |
| `run(Pointer<OrtSession>, {...})` → `List<Pointer<OrtValue>>`                                 | Run inference with FFI pointers                       |
| `releaseSession(Pointer<OrtSession>)`                                                         | Releases session resources                            |
| `releaseSessionOptions(Pointer<OrtSessionOptions>)`                                           | Releases options resources                            |
| `dispose()`                                                                                   | Cleans up all allocated resources                     |

#### `OrtTensor` Class

Low-level tensor helper that works with FFI pointers.

| Method                                                                 | Description            |
| ---------------------------------------------------------------------- | ---------------------- |
| `createFloat(List<int> shape, Float32List data)` → `Pointer<OrtValue>` | Create float32 tensor  |
| `createInt64(List<int> shape, List<int> data)` → `Pointer<OrtValue>`   | Create int64 tensor    |
| `getDataFloat(Pointer<OrtValue>, int elementCount)` → `Float32List`    | Read float tensor data |
| `release(Pointer<OrtValue>)`                                           | Release tensor memory  |

### Tensor Data Types

```dart
enum ONNXTensorElementDataType {
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,      // float32
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,      // uint8
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,       // int8
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,     // uint16
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,      // int16
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,      // int32
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,      // int64
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,     // string
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,      // bool
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,    // float16
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,     // float64
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,    // uint32
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,    // uint64
}
```

## 🔄 Regenerating Bindings

If you need to update the FFI bindings from a new version of the C API:

```bash
dart run ffigen --config ffigen.yaml
```

Configuration in `ffigen.yaml`:

```yaml
name: OnnxRuntimeBindings
description: Bindings for ONNX Runtime C API
output: lib/bindings/onnxruntime_generated.dart
headers:
  entry-points:
    - headers/onnxruntime_c_api.h
```

## 🧪 Running the Example

The example app demonstrates a complete MNIST digit recognition workflow:

1. **Model Loading**: Loads MNIST model from assets using high-level `OrtSessionWrapper`
2. **Provider Configuration**: Shows how to configure execution providers (commented out by default)
3. **Tensor Creation**: Creates synthetic input using `OrtValueWrapper.fromFloat()`
4. **Inference**: Runs inference with `session.runFloat()` returning pure Dart types
5. **Results**: Displays predicted digit and inference time

```bash
cd example
flutter pub get
flutter run
```

### Example Output

The app shows:

- **Status**: Initialization state
- **Model Info**: Auto-discovered inputs/outputs (1 input, 1 output for MNIST)
- **Inference Button**: Runs inference on a synthetic "3" digit pattern
- **Prediction Result**: Shows the predicted digit (0-9) and inference time in milliseconds

## 🛣️ Roadmap

- [x] Runtime and environment initialization
- [x] Inference session creation
- [x] Input/output tensor support
- [x] Inference execution
- [x] Execution provider support (CUDA, ROCm, CoreML)
- [x] Higher-level user-friendly APIs (without FFI pointers)
- [ ] Unit and integration tests

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Microsoft ONNX Runtime](https://onnxruntime.ai/) - Machine learning inference library
- [Flutter FFI](https://flutter.dev/to/ffi-package) - Flutter FFI documentation
- [ffigen](https://pub.dev/packages/ffigen) - Dart bindings generator

## 📞 Support

For bugs or feature requests, please open an issue on [GitHub](https://github.com/adrinator/flutter_ort_plugin/issues).

---

<p align="center">Developed with ❤️ using Flutter and ONNX Runtime</p>
