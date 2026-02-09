# flutter_ort_plugin

Flutter plugin for [ONNX Runtime](https://onnxruntime.ai/) inference via Dart FFI. Load `.onnx` models and run them natively on Android, iOS, macOS, Linux, and Windows.

ONNX Runtime version: **1.24.1**

## Installation

```yaml
dependencies:
  flutter_ort_plugin:
    git:
      url: https://github.com/adrinator/flutter_ort_plugin.git
```

## Platform setup

### iOS

- **Minimum version**: iOS **15.1**
- **Dependency**: CocoaPods `onnxruntime-c` (linked automatically by the plugin)

Recommended steps:

```bash
cd ios
pod install
```

If you run into linking/symbol issues when using `DynamicLibrary.process()` on iOS, ensure the plugin is properly registered in your app (Flutter plugin registrant) and do a clean build.

### Android

- **Minimum SDK**: Android **24** (`minSdk 24`)
- **Runtime**: `onnxruntime-android` **1.24.1** (pulled via Gradle by the plugin)
- **NDK**: required to build the plugin FFI target (see the NDK version configured in the plugin)

## Quick Start

```dart
import 'dart:typed_data';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

// 1. Initialize runtime (once)
final runtime = OnnxRuntime.instance;
runtime.initialize();
runtime.createEnvironment();

// 2. Load model (auto-selects best provider for the platform)
final session = OrtSessionWrapper.create('path/to/model.onnx');

// 3. Create input tensor
final input = OrtValueWrapper.fromFloat(
  runtime,
  [1, 3, 224, 224],  // shape
  Float32List(1 * 3 * 224 * 224),  // data
);

// 4. Run inference -> pure Dart output
final results = session.runFloat(
  {session.inputNames.first: input},
  [1000],  // output element count
);

final predictions = results.first; // Float32List

// 5. Cleanup
input.release();
session.dispose();
runtime.dispose();
```

## Avoid UI freezes (Isolate)

FFI calls are synchronous. Heavy model loading or inference can block the Flutter UI thread.

Use `OrtIsolateSession` to run everything in a background isolate:

```dart
final runtime = OnnxRuntime.instance;
runtime.initialize();
runtime.createEnvironment();

final session = await OrtIsolateSession.create(
  OrtIsolateSessionConfig(modelPath: 'path/to/model.onnx'),
);

final input = OrtIsolateInput(
  shape: [1, 1, 28, 28],
  data: Float32List(28 * 28),
);

final outputs = await session.runFloat(
  {session.inputNames.first: input},
  [10],
);

await session.dispose();
```

## Execution Providers

The plugin auto-detects the best provider per platform:

| Platform      | Default providers   | Supported   | Notes                                        |
| ------------- | ------------------- | ----------- | -------------------------------------------- |
| iOS/macOS     | CoreML, CPU         | ✅ Fully    | CoreML via dedicated config                  |
| Android       | XNNPACK, NNAPI, CPU | ✅ Fully    | NNAPI with flags, XNNPACK with thread config |
| Linux/Windows | CPU                 | ✅ CPU only | GPU providers via generic API only           |

### Provider Implementation Status

| Provider   | Status          | Notes                                            |
| ---------- | --------------- | ------------------------------------------------ |
| CPU        | ✅ Ready        | Always available, built-in                       |
| CoreML     | ✅ Ready        | iOS/macOS acceleration                           |
| NNAPI      | ✅ Ready        | Android NPU/GPU with FP16/NCHW flags             |
| XNNPACK    | ✅ Ready        | Android CPU SIMD optimization with thread config |
| QNN        | ⚠️ Generic only | Qualcomm - not fully implemented                 |
| CUDA       | ⚠️ Generic only | NVIDIA - not fully implemented                   |
| TensorRT   | ⚠️ Generic only | NVIDIA - not fully implemented                   |
| All others | ⚠️ Generic only | May work via generic API                         |

### Automatic (default)

```dart
// Providers are selected automatically
final session = OrtSessionWrapper.create('model.onnx');
```

### Manual

```dart
final session = OrtSessionWrapper.createWithProviders(
  'model.onnx',
  providers: [OrtProvider.coreML, OrtProvider.cpu],
  providerOptions: {
    OrtProvider.coreML: {'MLComputeUnits': 'ALL'},
  },
);
```

### XNNPACK (Android optimized CPU)

XNNPACK is the recommended provider for Android devices without a dedicated NPU:

```dart
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

final session = OrtSessionWrapper.createWithProviders(
  'model.onnx',
  providers: [OrtProvider.xnnpack, OrtProvider.cpu],
  providerOptions: {
    OrtProvider.xnnpack: XnnpackOptions(
      numThreads: 4,  // Use 4 threads (default: all cores)
    ).toMap(),
  },
);
```

### NNAPI (Android NPU/GPU)

NNAPI supports hardware acceleration but may have compatibility issues with some models:

```dart
final session = OrtSessionWrapper.createWithProviders(
  'model.onnx',
  providers: [OrtProvider.nnapi, OrtProvider.cpu],
  providerOptions: {
    OrtProvider.nnapi: {
      'use_fp16': 'true',      // Use FP16 for faster inference
      'use_nchw': 'false',     // Keep NHWC format
    },
  },
);
```

### Querying available providers

```dart
final providers = OrtProviders(OnnxRuntime.instance);

providers.getAvailableProviders();
// ['CoreMLExecutionProvider', 'CPUExecutionProvider']

providers.isProviderAvailable(OrtProvider.coreML); // true
```

## API Overview

### High-Level (no FFI pointers)

| Class               | Purpose                                              |
| ------------------- | ---------------------------------------------------- |
| `OrtSessionWrapper` | Load model, run inference, manage lifecycle          |
| `OrtValueWrapper`   | Create/read tensors with Dart types                  |
| `OrtProviders`      | Query and configure execution providers              |
| `OrtIsolateSession` | Run inference off the UI thread (background isolate) |

#### OrtSessionWrapper

```dart
// Auto providers
OrtSessionWrapper.create(modelPath);
OrtSessionWrapper.create(modelPath, providerOptions: { ... });

// Manual providers
OrtSessionWrapper.createWithProviders(modelPath, providers: [...]);

// Inference
session.run(inputs)         // -> List<OrtValueWrapper>
session.runFloat(inputs, outputSizes) // -> List<Float32List>

// Metadata
session.inputNames   // List<String>
session.outputNames  // List<String>

session.dispose();
```

#### OrtValueWrapper

```dart
// Create
OrtValueWrapper.fromFloat(runtime, shape, float32Data);
OrtValueWrapper.fromInt64(runtime, shape, int64Data);

// Read
value.toFloatList(elementCount); // -> Float32List

value.release();
```

### Low-Level (FFI pointers)

For advanced use cases, `OnnxRuntime` and `OrtTensor` expose the full C API with raw pointers. The generated bindings are also exported for direct access.

```dart
final rt = OnnxRuntime.instance;
final options = rt.createSessionOptions();
final session = rt.createSession('model.onnx', options);

final tensor = OrtTensor(rt);
final input = tensor.createFloat([1, 3], data);

final outputs = rt.run(session,
  inputNames: ['input'],
  inputValues: [input],
  outputNames: ['output'],
);

final result = tensor.getDataFloat(outputs.first, 10);

// Manual cleanup required
tensor.release(input);
for (final o in outputs) { tensor.release(o); }
rt.releaseSession(session);
rt.releaseSessionOptions(options);
```

## Example

The `example/` app demonstrates the high-level API using MNIST and is split into multiple pages:

- **Basic Inference**: load session + run inference
- **Execution Providers**: query/select providers
- **Isolate vs Sync**: visual UI-freeze comparison
- **Benchmark**: run N inferences and inspect statistics

```bash
cd example
flutter run
```

## Regenerating Bindings

```bash
dart run ffigen --config ffigen.yaml
```

## License

MIT. See [LICENSE](LICENSE).
