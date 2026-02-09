/// Configuration for ONNX Runtime session performance tuning.
///
/// These settings control threading, graph optimization, and execution mode.
/// They are applied to the session options before creating a session.
///
/// ## Android Big.LITTLE Recommendations
///
/// - **intraOpThreads**: 2–4 is optimal. Excess threads cause contention.
/// - **interOpThreads**: 1 for sequential mode, 2 for parallel mode.
/// - **graphOptimizationLevel**: 99 (all) for maximum optimization.
/// - **executionMode**: Sequential (0) is usually better on mobile.
class SessionConfig {
  /// Number of threads for intra-op parallelism (within a single node).
  /// 0 = ORT default, 1 = single-threaded, 2–4 recommended for mobile.
  final int intraOpThreads;

  /// Number of threads for inter-op parallelism (across nodes).
  /// Only effective when [executionMode] is [ExecutionMode.parallel].
  /// 0 = ORT default.
  final int interOpThreads;

  /// Graph optimization level.
  /// See [GraphOptLevel] for values.
  final GraphOptLevel graphOptimizationLevel;

  /// Execution mode: sequential or parallel.
  final ExecutionMode executionMode;

  const SessionConfig({
    this.intraOpThreads = 0,
    this.interOpThreads = 0,
    this.graphOptimizationLevel = GraphOptLevel.all,
    this.executionMode = ExecutionMode.sequential,
  });

  /// Recommended config for Android devices with Big.LITTLE architecture.
  static const SessionConfig androidOptimized = SessionConfig(
    intraOpThreads: 4,
    interOpThreads: 1,
    graphOptimizationLevel: GraphOptLevel.all,
    executionMode: ExecutionMode.sequential,
  );

  /// Default config (ORT defaults, no tuning).
  static const SessionConfig defaults = SessionConfig();
}

/// Graph optimization levels matching ORT's GraphOptimizationLevel enum.
enum GraphOptLevel {
  /// No graph optimizations.
  disabled(0, 'Disabled'),

  /// Basic optimizations (constant folding, redundant node elimination).
  basic(1, 'Basic'),

  /// Extended optimizations (node fusions, etc.).
  extended(2, 'Extended'),

  /// All optimizations including layout transforms.
  all(99, 'All (max)');

  final int value;
  final String label;
  const GraphOptLevel(this.value, this.label);
}

/// Execution mode for the session.
enum ExecutionMode {
  /// Nodes are executed sequentially.
  /// Better for mobile due to lower overhead.
  sequential(0, 'Sequential'),

  /// Nodes are executed in parallel when possible.
  /// May benefit multi-core desktop but adds overhead on mobile.
  parallel(1, 'Parallel');

  final int value;
  final String label;
  const ExecutionMode(this.value, this.label);
}
