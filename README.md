# flutter_ort_plugin

Flutter plugin for [ONNX Runtime](https://onnxruntime.ai/) inference via Dart FFI. Load `.onnx` models and run them natively on Android, iOS, and Linux.

ONNX Runtime version: **1.24.1**

## Platform Support

| Platform    | Minimum Version      | Execution Providers         | Status          |
| ----------- | -------------------- | --------------------------- | --------------- |
| **Android** | API 24 (Android 7.0) | WebGPU, NNAPI, XNNPACK, CPU | ✅ Full support |
| **iOS**     | iOS 15.1             | CoreML, CPU                 | ✅ Full support |
| **Linux**   | Any                  | CPU only                    | ✅ CPU support  |

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
- **Runtime**: Custom-compiled ONNX Runtime **1.24.1** from official repository (includes WebGPU/NNAPI/XNNPACK providers)
- **NDK**: required to build the plugin FFI target (see the NDK version configured in the plugin)

#### ONNX Runtime Build Strategies

The plugin supports two different ONNX Runtime builds for Android:

| Strategy      | Description                                    | Size    | When to Use                            |
| ------------- | ---------------------------------------------- | ------- | -------------------------------------- |
| **Standard**  | Basic CPU execution only                       | Smaller | Simple models, CPU-only inference      |
| **Providers** | Full provider support (WebGPU, NNAPI, XNNPACK) | Larger  | Performance-critical apps with GPU/NPU |

#### Building with Custom Strategy

By default, the plugin uses the **standard** build. To use the **providers** build with WebGPU/NNAPI/XNNPACK support:

```bash
# Build with providers (includes WebGPU, NNAPI, XNNPACK)
flutter build apk --android-project-arg=ORT_STRATEGY=providers

# Or for App Bundle
flutter build appbundle --android-project-arg=ORT_STRATEGY=providers

# For debug builds
flutter build apk --debug --android-project-arg=ORT_STRATEGY=providers
```

#### Provider Requirements

- **WebGPU**: Requires Android device with GPU support and Vulkan drivers
- **NNAPI**: Requires Android API 27+ for best compatibility
- **XNNPACK**: Works on all ARM devices (NEON SIMD)

The providers build is larger but enables hardware acceleration. Use the standard build for smaller app size or if you only need CPU inference.

#### Performance Considerations

**CPU vs Providers**: Many models actually perform better with CPU inference than with hardware providers, especially:

- Small to medium-sized models (<50MB)
- Models with many small operations
- Models not optimized for mobile GPUs/NPUs
- First-generation inference (warm-up overhead on providers)

**Recommendation**: Always test both strategies with your specific model:

```bash
# Test standard CPU build
flutter build apk --debug
# Run benchmarks with your model

# Test providers build
flutter build apk --debug --android-project-arg=ORT_STRATEGY=providers
# Run benchmarks with your model

# Compare inference latency and accuracy
```

The providers build shines with large models (>100MB) and operations well-suited for parallel GPU execution, but don't assume it's always faster.

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

| Platform | Default providers           | Supported   | Notes                                         |
| -------- | --------------------------- | ----------- | --------------------------------------------- |
| iOS      | CoreML, CPU                 | ✅ Fully    | CoreML via dedicated config                   |
| Android  | WebGPU, NNAPI, XNNPACK, CPU | ✅ Fully    | WebGPU via Dawn, NNAPI flags, XNNPACK threads |
| Linux    | CPU                         | ✅ CPU only | CPU execution provider                        |

> **Note**: Android providers (WebGPU, NNAPI, XNNPACK) require building with `--android-project-arg=ORT_STRATEGY=providers`. See Android setup section for details.

### Provider Implementation Status

| Provider | Status          | Platform | Notes                                     |
| -------- | --------------- | -------- | ----------------------------------------- |
| CPU      | ✅ Ready        | All      | Always available, built-in                |
| CoreML   | ✅ Ready        | iOS      | Apple Neural Engine/GPU acceleration      |
| WebGPU   | ✅ Ready        | Android  | GPU acceleration via Dawn/WebGPU support  |
| NNAPI    | ✅ Ready        | Android  | NPU/GPU with FP16/NCHW/CPU-disabled flags |
| XNNPACK  | ✅ Ready        | Android  | CPU SIMD optimization with thread config  |
| QNN      | ⚠️ Generic only | Android  | Qualcomm NPU via generic API              |

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

### WebGPU (Android GPU acceleration)

WebGPU provides hardware-accelerated inference on Android devices with GPU support:

```dart
final session = OrtSessionWrapper.createWithProviders(
  'model.onnx',
  providers: [OrtProvider.webGpu, OrtProvider.cpu],
  providerOptions: {
    // WebGPU options can be added here if needed
    OrtProvider.webGpu: {},
  },
);
```

### Querying available providers

```dart
final providers = OrtProviders(OnnxRuntime.instance);

providers.getAvailableProviders();
// ['WebGpuExecutionProvider', 'NnapiExecutionProvider', 'CPUExecutionProvider']

providers.isProviderAvailable(OrtProvider.webGpu); // true
```

## Performance Tuning

Fine-tune session options for optimal performance on your target device:

```dart
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

final session = OrtSessionWrapper.create(
  'model.onnx',
  sessionConfig: SessionConfig(
    intraOpThreads: 4,                    // Threads within ops (0 = ORT default)
    interOpThreads: 1,                    // Threads across ops (0 = ORT default)
    graphOptimizationLevel: GraphOptLevel.all, // Max graph optimizations
    executionMode: ExecutionMode.sequential,    // Better on mobile
  ),
);
```

### Android Big.LITTLE Optimization

For Android devices with heterogeneous cores, limit intra-op threads to avoid contention:

```dart
final session = OrtSessionWrapper.create(
  'model.onnx',
  sessionConfig: SessionConfig.androidOptimized, // Pre-configured for Android
);
```

### Available Options

| Option                   | Values                              | Description                           |
| ------------------------ | ----------------------------------- | ------------------------------------- |
| `intraOpThreads`         | `0` (auto) or integer               | Parallelism within a single operation |
| `interOpThreads`         | `0` (auto) or integer               | Parallelism across independent nodes  |
| `graphOptimizationLevel` | `disabled`/`basic`/`extended`/`all` | Graph transformation aggressiveness   |
| `executionMode`          | `sequential`/`parallel`             | Node execution order                  |

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

The `example/` app demonstrates real-world computer vision inference with YOLO models and includes comprehensive performance tuning:

- **YOLO Setup**: Model selection, provider configuration, and performance tuning UI
- **Camera Detection**: Real-time YOLO inference on camera feed with FPS/inference stats
- **Image Detection**: Static image inference with bounding box overlay
- **Video Detection**: Frame-by-frame inference on video with detection overlay
- **Performance Tuning**: Configure threading, graph optimization, and execution mode
- **Execution Providers**: Test different providers (WebGPU, NNAPI, XNNPACK, CoreML)

Features demonstrated:

- Dynamic model loading (.onnx/.ort formats)
- Platform-aware provider selection (WebGPU/NNAPI/XNNPACK on Android, CoreML on iOS)
- Session configuration for Android Big.LITTLE optimization
- Provider-specific options (NNAPI flags, XNNPACK threads, CoreML compute units)
- Background isolate inference to prevent UI freezes

```bash
cd example
flutter run
```

## Regenerating Bindings

```bash
dart run ffigen --config ffigen.yaml
```

## Recent Changes

### v1.0.3+

- **WebGPU Support**: Added WebGPU execution provider for Android GPU acceleration
- **Session Configuration**: New `SessionConfig` class for fine-tuning performance
  - Intra-op/inter-op thread control
  - Graph optimization levels (disabled → all)
  - Execution modes (sequential/parallel)
  - Android Big.LITTLE optimization preset
- **Performance Tuning UI**: Example app now includes comprehensive tuning controls
- **Video Detection**: Fixed playback stuttering with self-scheduling inference loop
- **Provider Summary**: Fixed provider options display to respect manual selection

### Provider Priority Updates

- Android now prioritizes GPU providers: WebGPU → NNAPI → XNNPACK → CPU
- iOS: CoreML → CPU
- Linux: CPU only

## License

MIT. See [LICENSE](LICENSE).
