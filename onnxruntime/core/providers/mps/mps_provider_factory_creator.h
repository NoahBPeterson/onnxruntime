#pragma once

#include <memory>

#include "core/providers/providers.h"

//#include "core/providers/mps/mps_execution_provider_factory.h"

namespace onnxruntime {

struct MPSProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create();
};

struct ProviderInfo_MPS {
  virtual int mpsGetDeviceCount() = 0;

  //virtual void cannMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
  //virtual void cannMemcpy_DeviceToHost(void* dst, const void* src, size_t count) = 0;
  virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory>
  CreateExecutionProviderFactory() = 0;
  virtual ~ProviderInfo_MPS() = default;
};


}  // namespace onnxruntime
