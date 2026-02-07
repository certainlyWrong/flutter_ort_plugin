# Flutter ONNX Runtime Plugin

[![Flutter Version](https://img.shields.io/badge/Flutter-%5E3.3.0-blue)](https://flutter.dev)
[![Dart Version](https://img.shields.io/badge/Dart-%5E3.10.8-blue)](https://dart.dev)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.24.1-green)](https://onnxruntime.ai)

A Flutter FFI (Foreign Function Interface) plugin for running ONNX (Open Neural Network Exchange) models directly in Flutter applications. This plugin uses the native ONNX Runtime C API library for machine learning inference on multiple platforms.

## 🚀 Features

- **ONNX Model Execution**: Load and execute ONNX models (.onnx) directly in Flutter
- **Multiplatform Support**: Android, iOS, macOS, Linux, and Windows
- **Native FFI API**: Direct integration with the ONNX Runtime C API via Dart FFI
- **Auto-generated Bindings**: Uses `ffigen` to generate Dart bindings from C headers
- **Resource Management**: High-level API for managing sessions, environments, and tensors

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

### Loading a Model

```dart
// Create session options
final sessionOptions = runtime.createSessionOptions();

// Load ONNX model
final session = runtime.createSession(
  'assets/model.onnx',  // Path to the .onnx file
  sessionOptions,
);
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
│   ├── onnx_runtinme.dart          # High-level API (OnnxRuntime)
│   ├── flutter_ort_plugin.dart     # Plugin exports
│   └── bindings/
│       └── onnxruntime_generated.dart  # Generated FFI bindings
├── src/                            # Native C/C++ code (if needed)
├── linux/                          # Linux CMake configuration
├── android/                        # Android Gradle configuration
├── ios/                            # iOS CocoaPods configuration
├── macos/                          # macOS configuration
├── windows/                        # Windows configuration
└── example/                        # Example application
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

### `OnnxRuntime` Class

#### Methods

| Method                                                                | Description                                           |
| --------------------------------------------------------------------- | ----------------------------------------------------- |
| `initialize({String? libraryPath})`                                   | Initializes the runtime by loading the native library |
| `createEnvironment({int logLevel, String logId})`                     | Creates the ONNX Runtime environment                  |
| `createSessionOptions()`                                              | Creates configuration options for sessions            |
| `createSession(String modelPath, Pointer<OrtSessionOptions> options)` | Creates a session from a model                        |
| `releaseSession(Pointer<OrtSession> session)`                         | Releases session resources                            |
| `releaseSessionOptions(Pointer<OrtSessionOptions> options)`           | Releases options resources                            |
| `dispose()`                                                           | Cleans up all allocated resources                     |

#### Properties

| Property        | Type               | Description                                   |
| --------------- | ------------------ | --------------------------------------------- |
| `isInitialized` | `bool`             | Indicates if the runtime has been initialized |
| `api`           | `Pointer<OrtApi>`  | Direct access to the ONNX Runtime API         |
| `environment`   | `Pointer<OrtEnv>?` | Pointer to the ONNX Runtime environment       |

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

```bash
cd example
flutter run
```

The example app demonstrates basic runtime initialization and displays the initialization status.

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
