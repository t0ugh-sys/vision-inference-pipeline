#include "backend_registry.hpp"
#include "encoder_interface.hpp"

#include <stdexcept>

#if defined(ENABLE_NVENC_ENCODER)
#include "backends/nvenc_encoder.hpp"
#endif
#if defined(ENABLE_MPP_ENCODER)
#include "backends/mpp_encoder.hpp"
#endif

EncoderBackendType detectAvailableEncoderBackend() {
#if defined(ENABLE_MPP_ENCODER)
  return EncoderBackendType::kRockchipMpp;
#elif defined(ENABLE_NVENC_ENCODER)
  return EncoderBackendType::kNvidiaNvEnc;
#else
  return EncoderBackendType::kCpu;
#endif
}

std::unique_ptr<IEncoderBackend> createEncoderBackend(EncoderBackendType type) {
  if (type == EncoderBackendType::kAuto) {
    type = detectAvailableEncoderBackend();
  }

  switch (type) {
#if defined(ENABLE_MPP_ENCODER)
    case EncoderBackendType::kRockchipMpp:
      return std::make_unique<MppEncoder>();
#endif
#if defined(ENABLE_NVENC_ENCODER)
    case EncoderBackendType::kNvidiaNvEnc:
      return std::make_unique<NvencEncoder>();
#endif
    case EncoderBackendType::kCpu:
    default:
      throw std::runtime_error(
          "Encoder backend '" + toString(type) +
          "' is not available in this build. Available: " + availableEncoderBackends());
  }
}
