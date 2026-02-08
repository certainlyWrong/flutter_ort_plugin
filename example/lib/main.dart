import 'dart:io';

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/material.dart';
import 'package:flutter_ort_plugin/flutter_ort_plugin.dart';

import 'benchmark_page.dart';
import 'inference_page.dart';
import 'isolate_page.dart';
import 'providers_page.dart';
import 'shared.dart';
import 'yolo_camera_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ONNX Runtime Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

// ---------------------------------------------------------------------------
// Home — runtime init + navigation to example pages
// ---------------------------------------------------------------------------

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _runtime = OnnxRuntime.instance;
  bool _initialized = false;
  String? _error;
  String _deviceInfo = '';
  String? _modelPath;
  String? _yoloModelPath;
  List<String> _availableProviders = [];
  List<OrtProvider> _defaultProviders = [];

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      _runtime.initialize();
      _runtime.createEnvironment(logLevel: 3, logId: 'ORT Demo');

      final providers = OrtProviders(_runtime);
      _availableProviders = providers.getAvailableProviders();
      _defaultProviders = providers.getDefaultProvidersForPlatform();

      await _collectDeviceInfo();
      _modelPath = await copyModelToTemp();
      _yoloModelPath = await copyYoloModelToTemp();

      setState(() => _initialized = true);
    } catch (e, stack) {
      debugPrint('Init error: $e\n$stack');
      setState(() => _error = e.toString());
    }
  }

  Future<void> _collectDeviceInfo() async {
    final plugin = DeviceInfoPlugin();
    try {
      if (Platform.isAndroid) {
        final info = await plugin.androidInfo;
        _deviceInfo =
            '${info.brand} ${info.model} — Android ${info.version.release}';
      } else if (Platform.isIOS) {
        final info = await plugin.iosInfo;
        _deviceInfo = '${info.utsname.machine} — iOS ${info.systemVersion}';
      } else if (Platform.isMacOS) {
        final info = await plugin.macOsInfo;
        _deviceInfo = '${info.model} — macOS ${info.osRelease}';
      } else if (Platform.isLinux) {
        final info = await plugin.linuxInfo;
        _deviceInfo = info.prettyName;
      } else if (Platform.isWindows) {
        final info = await plugin.windowsInfo;
        _deviceInfo = '${info.computerName} — ${info.productName}';
      }
    } catch (_) {
      _deviceInfo = Platform.operatingSystem;
    }
  }

  void _push(Widget page) {
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => page));
  }

  @override
  void dispose() {
    _runtime.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('ONNX Runtime Demo'),
        backgroundColor: cs.inversePrimary,
      ),
      body: !_initialized && _error == null
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? Center(child: ErrorCard(message: _error!))
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                SectionCard(
                  title: 'Runtime',
                  children: [
                    InfoRow('Device', _deviceInfo),
                    InfoRow('Platform', Platform.operatingSystem.toUpperCase()),
                    InfoRow(
                      'Available providers',
                      _availableProviders.join('\n'),
                    ),
                    InfoRow(
                      'Default providers',
                      _defaultProviders.map((p) => p.name).join(', '),
                    ),
                  ],
                ),
                const SizedBox(height: 24),
                Text(
                  'Examples',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                _NavTile(
                  icon: Icons.play_arrow,
                  color: Colors.blue,
                  title: 'Basic Inference',
                  subtitle:
                      'Load model with auto providers, select a digit, run inference with probability bars',
                  onTap: () => _push(InferencePage(modelPath: _modelPath!)),
                ),
                _NavTile(
                  icon: Icons.tune,
                  color: Colors.teal,
                  title: 'Execution Providers',
                  subtitle:
                      'Query available providers, check OrtProvider enum, create session with manual selection',
                  onTap: () => _push(ProvidersPage(modelPath: _modelPath!)),
                ),
                _NavTile(
                  icon: Icons.compare_arrows,
                  color: Colors.deepOrange,
                  title: 'Isolate vs Sync',
                  subtitle:
                      'Compare sync (freezes UI) vs isolate (smooth). Watch the spinner!',
                  onTap: () => _push(IsolatePage(modelPath: _modelPath!)),
                ),
                _NavTile(
                  icon: Icons.speed,
                  color: Colors.purple,
                  title: 'Benchmark',
                  subtitle:
                      'Run N inferences, see avg/median/p95/p99/stddev and histogram',
                  onTap: () => _push(BenchmarkPage(modelPath: _modelPath!)),
                ),
                _NavTile(
                  icon: Icons.camera_alt,
                  color: Colors.red,
                  title: 'YOLO Real-Time',
                  subtitle:
                      'YOLOv11 object detection with live camera feed and bounding boxes',
                  onTap: () =>
                      _push(YoloCameraPage(modelPath: _yoloModelPath!)),
                ),
              ],
            ),
    );
  }
}

class _NavTile extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _NavTile({
    required this.icon,
    required this.color,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: color.withValues(alpha: 0.15),
          child: Icon(icon, color: color),
        ),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
        subtitle: Text(subtitle, style: const TextStyle(fontSize: 12)),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}
