#pragma once

#include "onnxruntime_c_api.h"
//#include "core/framework/provider_options.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_MPS, _In_ OrtSessionOptions* options)
ORT_ALL_ARGS_NONNULL;

#ifdef __cplusplus
}
#endif
