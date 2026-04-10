#include "backend_registry.hpp"
#include "preproc_interface.hpp"
#include "backends/cuda_preprocessor.hpp"

#include <cstdlib>
#include <stdexcept>

#if defined(ENABLE_RGA_PREPROC)
#include "backends/rga_preprocessor.hpp"
#endif

PreprocBackendType detectAvailablePreprocBackend() {
#if defined(ENABLE_CUDA_PREPROC)
  return PreprocBackendType::kNvidiaCuda;
#elif defined(ENABLE_RGA_PREPROC)
  return PreprocBackendType::kRockchipRga;
#else
  return PreprocBackendType::kAuto;
#endif
}

std::unique_ptr<IPreprocessorBackend> createPreprocBackend(PreprocBackendType type) {
  if (type == PreprocBackendType::kAuto) {
    type = detectAvailablePreprocBackend();
  }

  switch (type) {
#if defined(ENABLE_RGA_PREPROC)
    case PreprocBackendType::kRockchipRga:
      return std::make_unique<RgaPreprocessor>();
#endif

#if defined(ENABLE_CUDA_PREPROC)
    case PreprocBackendType::kNvidiaCuda: {
      auto preproc = std::make_unique<CudaPreprocessor>();
      if (const char* gpu_id = std::getenv("CUDA_DEVICE")) {
        preproc->setGpuId(std::atoi(gpu_id));
      }
      return preproc;
    }
#endif

    case PreprocBackendType::kCpu:
    default:
      throw std::runtime_error(
          "Preprocessor backend '" + toString(type) +
          "' is not available in this build. Available: " + availablePreprocBackends());
  }
}
