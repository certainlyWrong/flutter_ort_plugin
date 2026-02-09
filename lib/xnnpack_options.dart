import 'dart:io';

/// Configuration options for the XNNPACK execution provider.
///
/// XNNPACK is a highly optimized library for neural network inference on ARM
/// and x86 CPUs. It uses SIMD instructions (NEON on ARM, SSE/AVX on x86) for
/// maximum throughput.
///
/// ```dart
/// OrtSessionWrapper.createWithProviders(
///   modelPath,
///   providers: [OrtProvider.xnnpack, OrtProvider.cpu],
///   providerOptions: {
///     OrtProvider.xnnpack: XnnpackOptions(numThreads: 4).toMap(),
///   },
/// );
/// ```
class XnnpackOptions {
  /// Number of threads for XNNPACK's internal thread pool.
  ///
  /// - `0` (default): uses the session-level `intra_op_num_threads` setting.
  /// - `> 0`: overrides with the specified thread count.
  ///
  /// If not explicitly set and no user override is provided, defaults to
  /// [Platform.numberOfProcessors] for optimal performance on the device.
  final int numThreads;

  const XnnpackOptions({this.numThreads = 0});

  /// Creates [XnnpackOptions] from a provider options map.
  ///
  /// If `intra_op_num_threads` is not present in the map, automatically
  /// sets it to [Platform.numberOfProcessors] for best performance.
  factory XnnpackOptions.fromMap(Map<String, String> map) {
    final threads = map['intra_op_num_threads'];
    if (threads != null) {
      return XnnpackOptions(numThreads: int.tryParse(threads) ?? 0);
    }
    // Auto-configure: use all available cores
    return XnnpackOptions(numThreads: Platform.numberOfProcessors);
  }

  /// Converts to the key-value map expected by the ONNX Runtime generic API.
  Map<String, String> toMap() {
    if (numThreads <= 0) return const {};
    return {'intra_op_num_threads': numThreads.toString()};
  }

  @override
  String toString() => 'XnnpackOptions(numThreads: $numThreads)';
}
