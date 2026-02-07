Pod::Spec.new do |s|
  s.name             = 'flutter_ort_plugin'
  s.version          = '0.0.1'
  # ... outras configurações ...

  # Adicione esta linha:
  s.dependency 'onnxruntime-c', '1.24.1'
  
  s.platform = :ios, '12.0' # ORT exige pelo menos iOS 11 ou 12
end