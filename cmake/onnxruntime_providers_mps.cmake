set(mps_required_libs "")
if(APPLE)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
  list(APPEND mps_required_libs ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK} ${MPS_FRAMEWORK})
  add_compile_definitions(USE_MPS=1)
endif()

set(mps_src_patterns
  "${ONNXRUNTIME_ROOT}/core/providers/mps/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/mps/*.cc"
)

file(GLOB mps_src CONFIGURE_DEPENDS ${mps_src_patterns})

source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${mps_src})

onnxruntime_add_static_library(onnxruntime_providers_mps ${mps_src})

if(APPLE)
  target_include_directories(onnxruntime_providers_mps PRIVATE
    ${ONNXRUNTIME_ROOT}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ONNXRUNTIME_ROOT}/core/providers/mps
    ${CMAKE_CURRENT_BINARY_DIR}/onnx
    ${REPO_ROOT}/cmake/external/onnx
    flatbuffers::flatbuffers
  )

  target_link_libraries(onnxruntime_providers_mps PRIVATE
    onnxruntime_common
    onnxruntime_framework
    onnxruntime_providers
    onnx
    flatbuffers
    ${mps_required_libs}
  )

  set_target_properties(onnxruntime_providers_mps PROPERTIES
    FOLDER "ONNXRuntime"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )

  install(TARGETS onnxruntime_providers_mps
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
