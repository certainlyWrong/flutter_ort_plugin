import 'package:flutter/material.dart';
import 'package:flutter_ort_plugin/onnx_runtinme.dart';

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
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'ONNX Runtime Flutter Plugin'),
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
  String _status = 'Not initialized';
  final _runtime = OnnxRuntime.instance;

  @override
  void initState() {
    super.initState();
    _initializeRuntime();
  }

  Future<void> _initializeRuntime() async {
    try {
      _runtime.initialize();
      _runtime.createEnvironment(logLevel: 3, logId: 'OnnxRuntimeExample');
      setState(() {
        _status = 'Initialized successfully';
      });
    } catch (e) {
      setState(() {
        _status = 'Error: $e';
      });
    }
  }

  @override
  void dispose() {
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
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            SelectableText(
              'ONNX Runtime Status:',
              style: TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 16),
            SelectableText(
              _status,
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            if (_runtime.isInitialized)
              const Icon(Icons.check_circle, color: Colors.green, size: 64)
            else
              const CircularProgressIndicator(),
          ],
        ),
      ),
    );
  }
}
