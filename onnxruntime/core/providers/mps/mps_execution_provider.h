#pragma once

#include "core/framework/execution_provider.h"
//#include "core/framework/kernel_registry.h"
//#include "core/platform/ort_mutex.h"
//#include "core/graph/constants.h"

namespace onnxruntime {

// Forward declarations
class MPSExecutionProvider;
class IKernelLookup;

// Configuration options for the MPS EP
struct MPSProviderOptions {
  bool enable_cpu_fallback{true};  // Allow fallback to CPU for unsupported operators
};

// State information to be passed to compute of kernel implementation
struct MPSFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocator_handle = nullptr;
  std::string node_name;
};

// MPS Execution Provider
class MPSExecutionProvider : public IExecutionProvider {
 public:
  explicit MPSExecutionProvider(const MPSProviderOptions& options = MPSProviderOptions());
  ~MPSExecutionProvider() override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& kernel_lookup) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                        std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;  // TODO: Return Metal device/command queue if needed
  }

 private:
  MPSProviderOptions options_;
  std::shared_ptr<KernelRegistry> kernel_registry_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MPSExecutionProvider);
};

}  // namespace onnxruntime
