#import "OnnxLinkerFix.h"
#import <onnxruntime/onnxruntime_c_api.h>

@implementation FlutterOrtPlugin

+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
    // Force the linker to include OrtGetApiBase in the final binary.
    // Without this reference, the static onnxruntime library symbols
    // would be dead-stripped since nothing in ObjC/Swift code uses them.
    __unused const OrtApiBase* api = OrtGetApiBase();
}

@end
