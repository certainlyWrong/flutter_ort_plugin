Pod::Spec.new do |s|
  s.name             = 'flutter_ort_plugin'
  s.version          = '0.0.1'
  s.summary          = 'Plugin ONNX Runtime via CocoaPods'
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.static_framework = true

  s.dependency 'Flutter'
  s.dependency 'onnxruntime-c', '1.24.1'

  s.platform = :ios, '15.1'
  s.frameworks = 'Accelerate', 'CoreML'
  s.libraries = 'c++'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'CLANG_ALLOW_NON_MODULAR_INCLUDES_IN_FRAMEWORK_MODULES' => 'YES'
  }
end