// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/mps/mps_provider_factory.h"
#include "core/providers/mps/mps_provider_factory_creator.h"
#include "core/providers/mps/mps_execution_provider.h"

//#import <Foundation/Foundation.h>
//#import <Metal/Metal.h>


namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct MPSProviderFactory : IExecutionProviderFactory {
  MPSProviderFactory() = default;
  ~MPSProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> MPSProviderFactory::CreateProvider() {
  return std::make_unique<MPSExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> MPSProviderFactoryCreator::Create() {
  return std::make_shared<MPSProviderFactory>();
}

struct ProviderInfo_MPS_Impl : ProviderInfo_MPS {
  int mpsGetDeviceCount() override {
    /*
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
    */
      return 1;
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory() override {
    return std::make_shared<MPSProviderFactory>();
  }
} g_info;

struct MPS_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* /*void_params*/) override {
    return std::make_shared<MPSProviderFactory>();
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }
} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}

}
