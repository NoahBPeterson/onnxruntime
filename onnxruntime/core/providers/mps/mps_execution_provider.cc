#include "core/providers/mps/mps_execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/allocator.h"
#include "core/common/logging/logging.h"
#include "core/session/onnxruntime_cxx_api.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace onnxruntime {

namespace {
auto& GetMPSLogger() {
  static auto logger = logging::LoggingManager::DefaultLogger();
  return logger;
}
}  // namespace

MPSExecutionProvider::MPSExecutionProvider(const MPSProviderOptions& options)
    : IExecutionProvider{kMPSExecutionProvider},
      options_(options) {
  LOGS(GetMPSLogger(), INFO) << "Creating MPS Execution Provider";

  bool has_metal = false;
#if defined(__APPLE__)
  #if TARGET_OS_OSX || TARGET_OS_IPHONE
    has_metal = true;
    LOGS(GetMPSLogger(), INFO) << "Metal is available on this platform";
  #else
    LOGS(GetMPSLogger(), WARNING) << "Metal is not available on this platform";
  #endif
#else
  LOGS(GetMPSLogger(), WARNING) << "Not running on Apple platform";
#endif

  if (!has_metal) {
    LOGS(GetMPSLogger(), ERROR) << "MPS Execution Provider requires Metal support";
    return;
  }

  // Initialize kernel registry
  kernel_registry_ = std::make_shared<KernelRegistry>();
  LOGS(GetMPSLogger(), INFO) << "Initialized kernel registry";
}

MPSExecutionProvider::~MPSExecutionProvider() {
  LOGS(GetMPSLogger(), INFO) << "Destroying MPS Execution Provider";
}

std::vector<std::unique_ptr<ComputeCapability>> MPSExecutionProvider::GetCapability(
    const GraphViewer& graph,
    const IKernelLookup& kernel_lookup) const {
  InlinedVector<NodeIndex> candidates;
  const logging::Logger& logger = *GetLogger();

  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      if (node.GetExecutionProviderType() == kMPSExecutionProvider) {
        candidates.push_back(node.Index());
      }
      continue;
    }

    const KernelCreateInfo* mps_kernel_def = kernel_lookup.LookUpKernel(node);
    // none of the provided registries has an MPS kernel for this node
    if (mps_kernel_def == nullptr) {
      LOGS(logger, INFO) << "MPS kernel not found in registries for Op type: " << node.OpType()
                        << " node name: " << node.Name();
      continue;
    }

    // For now, since we haven't implemented any kernels, we'll log this
    LOGS(logger, INFO) << "Found potential MPS kernel for Op type: " << node.OpType()
                      << " node name: " << node.Name()
                      << " (Note: Kernels not yet implemented)";

    // Add to candidates - we'll implement actual kernel support later
    candidates.push_back(node.Index());
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    auto sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  LOGS(logger, INFO) << "MPS GetCapability found " << result.size() << " supported nodes";
  return result;
}

common::Status MPSExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                                           std::vector<NodeComputeInfo>& node_compute_funcs) {
  LOGS(GetMPSLogger(), INFO) << "Compiling " << fused_nodes.size() << " fused nodes";

  for (const auto& fused_node_graph : fused_nodes) {
    // const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    LOGS(GetMPSLogger(), INFO) << "Processing fused node: " << fused_node.Name();

    NodeComputeInfo compute_info;

    // Set up state creation function
    compute_info.create_state_func = [](ComputeContext* context, FunctionState* state) {
      auto* p = new MPSFuncState();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, context->node_name};
      *state = p;
      return 0;
    };

    // Set up state release function
    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<MPSFuncState*>(state);
    };

    // Set up compute function
    compute_info.compute_func = [](FunctionState state, const OrtApi* /* api */, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);
      // MPSFuncState* mps_state = reinterpret_cast<MPSFuncState*>(state);

      // TODO: Implement actual compute logic using Metal Performance Shaders
      // This will involve:
      // 1. Getting input tensors from context
      // 2. Creating MPS graph/command buffer
      // 3. Executing the operation
      // 4. Writing results to output tensors

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> MPSExecutionProvider::GetKernelRegistry() const {
  LOGS(GetMPSLogger(), INFO) << "Returning kernel registry";
  return kernel_registry_;
}

std::unique_ptr<IDataTransfer> MPSExecutionProvider::GetDataTransfer() const {
  LOGS(GetMPSLogger(), INFO) << "Creating data transfer object";
  // TODO: Implement proper data transfer
  return nullptr;
}

}  // namespace onnxruntime
