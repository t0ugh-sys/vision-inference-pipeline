#include "backend_registry.hpp"
#include "decoder_interface.hpp"
#include "backends/nvdec_decoder.hpp"

#if defined(ENABLE_MPP_DECODER)
#include "backends/mpp_decoder.hpp"
#endif

#include <cstdlib>
#include <stdexcept>

DecoderBackendType detectAvailableDecoderBackend() {
#if defined(ENABLE_NVDEC_DECODER)
  return DecoderBackendType::kNvidiaNvdec;
#elif defined(ENABLE_MPP_DECODER)
  return DecoderBackendType::kRockchipMpp;
#else
  return DecoderBackendType::kAuto;
#endif
}

std::unique_ptr<IDecoderBackend> createDecoderBackend(DecoderBackendType type) {
  if (type == DecoderBackendType::kAuto) {
    type = detectAvailableDecoderBackend();
  }

  switch (type) {
#if defined(ENABLE_NVDEC_DECODER)
    case DecoderBackendType::kNvidiaNvdec: {
      auto decoder = std::make_unique<NvdecDecoder>();
      if (const char* gpu_id = std::getenv("CUDA_DEVICE")) {
        decoder->setGpuId(std::atoi(gpu_id));
      }
      return decoder;
    }
#endif

#if defined(ENABLE_MPP_DECODER)
    case DecoderBackendType::kRockchipMpp:
      return std::make_unique<MppDecoder>();
#endif

    case DecoderBackendType::kCpu:
    default:
      throw std::runtime_error(
          "Decoder backend '" + toString(type) +
          "' is not available in this build. Available: " + availableDecoderBackends());
  }
}
