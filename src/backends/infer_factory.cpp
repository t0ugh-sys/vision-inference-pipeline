#include "backend_registry.hpp"
#include "infer_interface.hpp"

#include <cstdlib>
#include <stdexcept>

#if defined(ENABLE_RKNN_INFER)
#include "backends/rknn_infer.hpp"
#endif

#if defined(ENABLE_TRT_INFER)
#include "backends/trt_infer.hpp"
#endif

InferBackendType detectAvailableInferBackend() {
#if defined(ENABLE_TRT_INFER)
  return InferBackendType::kNvidiaTrt;
#elif defined(ENABLE_RKNN_INFER)
  return InferBackendType::kRockchipRknn;
#else
  return InferBackendType::kAuto;
#endif
}

std::unique_ptr<IInferenceBackend> createInferBackend(InferBackendType type) {
  if (type == InferBackendType::kAuto) {
    type = detectAvailableInferBackend();
  }

  switch (type) {
#if defined(ENABLE_RKNN_INFER)
    case InferBackendType::kRockchipRknn:
      return std::make_unique<RknnInfer>();
#endif

#if defined(ENABLE_TRT_INFER)
    case InferBackendType::kNvidiaTrt: {
      auto infer = std::make_unique<TrtInfer>();
      if (const char* gpu_id = std::getenv("CUDA_DEVICE")) {
        infer->setGpuId(std::atoi(gpu_id));
      }
      return infer;
    }
#endif

    case InferBackendType::kOnnxRuntime:
    default:
      throw std::runtime_error(
          "Inference backend '" + toString(type) +
          "' is not available in this build. Available: " + availableInferBackends());
  }
}
