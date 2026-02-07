import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ONNX Runtime MNIST Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'ONNX Runtime MNIST'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _status = 'Initializing...';
  String _modelInfo = '';
  String _deviceInfo = '';
  String _gpuInfo = '';
  int? _predictedDigit;
  double _inferenceTimeMs = 0;
  bool _isLoading = true;
  bool _isRunning = false;

  final _runtime = OnnxRuntime.instance;
  final _deviceInfoPlugin = DeviceInfoPlugin();
  OrtSessionWrapper? _session;

  @override
  void initState() {
    super.initState();
    _initializeAndLoadModel();
  }

  Future<void> _initializeAndLoadModel() async {
    try {
      _runtime.initialize();
      _runtime.createEnvironment(logLevel: 3, logId: 'MNIST Demo');

      // Collect device and GPU info using high-level APIs
      await _collectDeviceInfo();

      // Copy model from assets to temp directory for loading
      final modelBytes = await rootBundle.load('assets/mnist/mnist-12.onnx');
      final tempDir = await getTemporaryDirectory();
      final modelFile = File('${tempDir.path}/mnist-12.onnx');
      await modelFile.writeAsBytes(modelBytes.buffer.asUint8List());

      // Load model using high-level API with GPU detection
      _session = OrtSessionWrapper.create(
        modelFile.path,
        configureProviders: (providers, options) {
          _configureExecutionProviders(providers, options);
        },
      );

      final inputNames = _session!.inputNames;
      final outputNames = _session!.outputNames;

      setState(() {
        _modelInfo =
            'Model: mnist-12.onnx\n'
            'Inputs: ${inputNames.join(", ")} (${_session!.inputCount})\n'
            'Outputs: ${outputNames.join(", ")} (${_session!.outputCount})';
        _status = 'Ready';
        _isLoading = false;
      });
    } catch (e, stack) {
      setState(() {
        _status = 'Error: $e';
        _isLoading = false;
      });
      debugPrint('Stack trace: $stack');
    }
  }

  Future<void> _collectDeviceInfo() async {
    final platform = Platform.operatingSystem;
    String deviceDetails = '';

    try {
      if (Platform.isAndroid) {
        final androidInfo = await _deviceInfoPlugin.androidInfo;
        deviceDetails =
            '${androidInfo.brand} ${androidInfo.model}\n'
            'Android ${androidInfo.version.release}\n'
            'SDK ${androidInfo.version.sdkInt}';
      } else if (Platform.isIOS) {
        final iosInfo = await _deviceInfoPlugin.iosInfo;
        deviceDetails =
            '${iosInfo.name} ${iosInfo.model}\n'
            'iOS ${iosInfo.systemVersion}';
      } else if (Platform.isLinux) {
        final linuxInfo = await _deviceInfoPlugin.linuxInfo;
        deviceDetails =
            '${linuxInfo.prettyName}\n'
            '${linuxInfo.machineId?.substring(0, 8) ?? 'Unknown'}';
      } else if (Platform.isMacOS) {
        final macInfo = await _deviceInfoPlugin.macOsInfo;
        deviceDetails =
            '${macInfo.computerName}\n'
            '${macInfo.osRelease}';
      } else if (Platform.isWindows) {
        final windowsInfo = await _deviceInfoPlugin.windowsInfo;
        deviceDetails =
            '${windowsInfo.computerName}\n'
            '${windowsInfo.editionId}';
      }
    } catch (e) {
      deviceDetails = 'Unable to get device details: $e';
    }

    setState(() {
      _deviceInfo = 'Platform: ${platform.toUpperCase()}\n$deviceDetails';
    });
  }

  void _configureExecutionProviders(
    OrtProviders providers,
    Pointer<OrtSessionOptions> options,
  ) {
    String gpuStatus = '';

    try {
      // Try GPU providers based on platform using high-level OrtProviders API
      if (Platform.isAndroid) {
        // Try GPU delegate for Android (TFLite-style or QNN)
        try {
          providers.appendExecutionProvider(
            options,
            'QNN',
            providerOptions: {'backend_path': 'libQnnCpu.so'},
          );
          gpuStatus = 'QNN (Qualcomm AI)';
        } catch (e) {
          gpuStatus = 'CPU only (QNN not available)';
        }
      } else if (Platform.isIOS) {
        // CoreML for Apple devices (Neural Engine / GPU / CPU)
        try {
          providers.appendCoreML(
            options,
            providerOptions: {'MLComputeUnits': 'All'},
          );
          gpuStatus = 'CoreML (Neural Engine/GPU)';
        } catch (e) {
          gpuStatus = 'CPU only (CoreML not available)';
        }
      } else if (Platform.isLinux || Platform.isWindows) {
        // Try CUDA for NVIDIA GPUs
        try {
          providers.appendCuda(options, deviceId: 0);
          gpuStatus = 'CUDA (NVIDIA GPU)';
        } catch (e) {
          // Try ROCm for AMD GPUs on Linux
          if (Platform.isLinux) {
            try {
              providers.appendROCm(options, deviceId: 0);
              gpuStatus = 'ROCm (AMD GPU)';
            } catch (e2) {
              gpuStatus = 'CPU only (no GPU detected)';
            }
          } else {
            gpuStatus = 'CPU only (no GPU detected)';
          }
        }
      } else if (Platform.isMacOS) {
        // CoreML for macOS
        try {
          providers.appendCoreML(
            options,
            providerOptions: {'MLComputeUnits': 'All'},
          );
          gpuStatus = 'CoreML (Neural Engine/GPU)';
        } catch (e) {
          gpuStatus = 'CPU only (CoreML not available)';
        }
      }
    } catch (e) {
      gpuStatus = 'CPU only (error: $e)';
    }

    setState(() {
      _gpuInfo = 'Execution Provider: $gpuStatus';
    });
  }

  Future<void> _runInference() async {
    if (_session == null) return;

    setState(() {
      _isRunning = true;
      _predictedDigit = null;
    });

    try {
      final stopwatch = Stopwatch()..start();

      // Create a synthetic 28x28 MNIST-like input (normalized 0-1)
      // Simulating a "3" digit pattern
      final inputData = _createDigitThreePattern();

      final inputValue = OrtValueWrapper.fromFloat(
        _runtime,
        [1, 1, 28, 28], // batch=1, channels=1, height=28, width=28
        inputData,
      );

      final inputs = {_session!.inputNames.first: inputValue};

      // Run inference - output is 10 floats (probabilities for digits 0-9)
      final results = _session!.runFloat(inputs, [10]);

      stopwatch.stop();

      // Find the digit with highest probability
      final output = results.first;
      int maxIndex = 0;
      double maxProb = output[0];
      for (var i = 1; i < output.length; i++) {
        if (output[i] > maxProb) {
          maxProb = output[i];
          maxIndex = i;
        }
      }

      inputValue.release();

      setState(() {
        _predictedDigit = maxIndex;
        _inferenceTimeMs = stopwatch.elapsedMilliseconds.toDouble();
        _isRunning = false;
      });
    } catch (e, stack) {
      setState(() {
        _status = 'Inference error: $e';
        _isRunning = false;
      });
      debugPrint('Stack trace: $stack');
    }
  }

  Float32List _createDigitThreePattern() {
    // Create a simple 28x28 image simulating the digit "3"
    final data = Float32List(28 * 28);

    // Initialize with zeros (background)
    for (var i = 0; i < data.length; i++) {
      data[i] = 0.0;
    }

    // Draw a simple "3" pattern with white pixels (1.0)
    final centerY = 14;
    final strokeWidth = 3;

    for (var x = 8; x < 20; x++) {
      // Top curve
      for (var y = centerY - 6; y < centerY - 6 + strokeWidth; y++) {
        if (x > 10) data[y * 28 + x] = 1.0;
      }
      // Middle curve
      for (var y = centerY - 1; y < centerY - 1 + strokeWidth; y++) {
        data[y * 28 + x] = 1.0;
      }
      // Bottom curve
      for (var y = centerY + 4; y < centerY + 4 + strokeWidth; y++) {
        if (x > 10) data[y * 28 + x] = 1.0;
      }
    }

    // Left side connectors
    for (var y = centerY - 6; y < centerY + 7; y++) {
      if ((y - (centerY - 6)) % 5 < strokeWidth) {
        data[y * 28 + 9] = 1.0;
      }
    }

    return data;
  }

  @override
  void dispose() {
    _session?.dispose();
    _runtime.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Status',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey,
                        ),
                      ),
                      const SizedBox(height: 8),
                      SelectableText(
                        _status,
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Model Info',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey,
                        ),
                      ),
                      const SizedBox(height: 8),
                      SelectableText(
                        _modelInfo.isEmpty ? 'Loading...' : _modelInfo,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Device Info',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey,
                        ),
                      ),
                      const SizedBox(height: 8),
                      SelectableText(
                        _deviceInfo.isEmpty ? 'Loading...' : _deviceInfo,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'GPU / Execution Provider',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey,
                        ),
                      ),
                      const SizedBox(height: 8),
                      SelectableText(
                        _gpuInfo.isEmpty ? 'Detecting...' : _gpuInfo,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color:
                              _gpuInfo.contains('GPU') ||
                                  _gpuInfo.contains('CoreML') ||
                                  _gpuInfo.contains('QNN')
                              ? Colors.green[700]
                              : Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
              if (!_isLoading)
                Center(
                  child: ElevatedButton.icon(
                    onPressed: _isRunning ? null : _runInference,
                    icon: _isRunning
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.play_arrow),
                    label: Text(_isRunning ? 'Running...' : 'Run Inference'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 32,
                        vertical: 16,
                      ),
                    ),
                  ),
                ),
              const SizedBox(height: 24),
              if (_predictedDigit != null)
                Center(
                  child: Card(
                    color: Theme.of(context).colorScheme.primaryContainer,
                    child: Padding(
                      padding: const EdgeInsets.all(24.0),
                      child: Column(
                        children: [
                          const Text(
                            'Predicted Digit',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            '$_predictedDigit',
                            style: const TextStyle(
                              fontSize: 72,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Inference time: ${_inferenceTimeMs.toStringAsFixed(1)} ms',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
