#include "backend_registry.hpp"

#include <vector>

namespace {

template <typename T>
std::string joinAvailable(const std::vector<T>& backends) {
  if (backends.empty()) {
    return "none";
  }

  std::string result;
  for (std::size_t index = 0; index < backends.size(); ++index) {
    if (index > 0) {
      result += ", ";
    }
    result += toString(backends[index]);
  }
  return result;
}

}  // namespace

std::string toString(DecoderBackendType type) {
  switch (type) {
    case DecoderBackendType::kAuto:
      return "auto";
    case DecoderBackendType::kRockchipMpp:
      return "rockchip-mpp";
    case DecoderBackendType::kNvidiaNvdec:
      return "nvidia-nvdec";
    case DecoderBackendType::kCpu:
      return "cpu";
    default:
      return "unknown";
  }
}

std::string toString(PreprocBackendType type) {
  switch (type) {
    case PreprocBackendType::kAuto:
      return "auto";
    case PreprocBackendType::kRockchipRga:
      return "rockchip-rga";
    case PreprocBackendType::kNvidiaCuda:
      return "nvidia-cuda";
    case PreprocBackendType::kCpu:
      return "cpu";
    default:
      return "unknown";
  }
}

std::string toString(InferBackendType type) {
  switch (type) {
    case InferBackendType::kAuto:
      return "auto";
    case InferBackendType::kRockchipRknn:
      return "rockchip-rknn";
    case InferBackendType::kNvidiaTrt:
      return "nvidia-tensorrt";
    case InferBackendType::kOnnxRuntime:
      return "onnxruntime";
    default:
      return "unknown";
  }
}

std::string toString(PostprocBackendType type) {
  switch (type) {
    case PostprocBackendType::kAuto:
      return "auto";
    case PostprocBackendType::kYoloV8:
      return "yolov8";
    case PostprocBackendType::kYolo26:
      return "yolo26";
    case PostprocBackendType::kYoloV5:
      return "yolov5";
    default:
      return "unknown";
  }
}

std::string toString(EncoderBackendType type) {
  switch (type) {
    case EncoderBackendType::kAuto:
      return "auto";
    case EncoderBackendType::kNvidiaNvEnc:
      return "nvidia-nvenc";
    case EncoderBackendType::kRockchipMpp:
      return "rockchip-mpp";
    case EncoderBackendType::kCpu:
      return "cpu";
    default:
      return "unknown";
  }
}

bool isCompiledIn(DecoderBackendType type) {
  switch (type) {
    case DecoderBackendType::kAuto:
      return true;
#if defined(ENABLE_MPP_DECODER)
    case DecoderBackendType::kRockchipMpp:
      return true;
#endif
#if defined(ENABLE_NVDEC_DECODER)
    case DecoderBackendType::kNvidiaNvdec:
      return true;
#endif
    case DecoderBackendType::kCpu:
      return false;
    default:
      return false;
  }
}

bool isCompiledIn(PreprocBackendType type) {
  switch (type) {
    case PreprocBackendType::kAuto:
      return true;
#if defined(ENABLE_RGA_PREPROC)
    case PreprocBackendType::kRockchipRga:
      return true;
#endif
#if defined(ENABLE_CUDA_PREPROC)
    case PreprocBackendType::kNvidiaCuda:
      return true;
#endif
    case PreprocBackendType::kCpu:
      return false;
    default:
      return false;
  }
}

bool isCompiledIn(InferBackendType type) {
  switch (type) {
    case InferBackendType::kAuto:
      return true;
#if defined(ENABLE_RKNN_INFER)
    case InferBackendType::kRockchipRknn:
      return true;
#endif
#if defined(ENABLE_TRT_INFER)
    case InferBackendType::kNvidiaTrt:
      return true;
#endif
    case InferBackendType::kOnnxRuntime:
      return false;
    default:
      return false;
  }
}

bool isCompiledIn(PostprocBackendType type) {
  switch (type) {
    case PostprocBackendType::kAuto:
    case PostprocBackendType::kYoloV8:
    case PostprocBackendType::kYolo26:
    case PostprocBackendType::kYoloV5:
      return true;
    default:
      return false;
  }
}

bool isCompiledIn(EncoderBackendType type) {
  switch (type) {
    case EncoderBackendType::kAuto:
      return true;
#if defined(ENABLE_MPP_ENCODER)
    case EncoderBackendType::kRockchipMpp:
      return true;
#endif
#if defined(ENABLE_NVENC_ENCODER)
    case EncoderBackendType::kNvidiaNvEnc:
      return true;
#endif
    case EncoderBackendType::kCpu:
      return false;
    default:
      return false;
  }
}

std::string availableDecoderBackends() {
  std::vector<DecoderBackendType> backends;
#if defined(ENABLE_MPP_DECODER)
  backends.push_back(DecoderBackendType::kRockchipMpp);
#endif
#if defined(ENABLE_NVDEC_DECODER)
  backends.push_back(DecoderBackendType::kNvidiaNvdec);
#endif
  return joinAvailable(backends);
}

std::string availablePreprocBackends() {
  std::vector<PreprocBackendType> backends;
#if defined(ENABLE_RGA_PREPROC)
  backends.push_back(PreprocBackendType::kRockchipRga);
#endif
#if defined(ENABLE_CUDA_PREPROC)
  backends.push_back(PreprocBackendType::kNvidiaCuda);
#endif
  return joinAvailable(backends);
}

std::string availableInferBackends() {
  std::vector<InferBackendType> backends;
#if defined(ENABLE_RKNN_INFER)
  backends.push_back(InferBackendType::kRockchipRknn);
#endif
#if defined(ENABLE_TRT_INFER)
  backends.push_back(InferBackendType::kNvidiaTrt);
#endif
  return joinAvailable(backends);
}

std::string availablePostprocBackends() {
  return "yolov8, yolo26, yolov5";
}

std::string availableEncoderBackends() {
  std::vector<EncoderBackendType> backends;
#if defined(ENABLE_MPP_ENCODER)
  backends.push_back(EncoderBackendType::kRockchipMpp);
#endif
#if defined(ENABLE_NVENC_ENCODER)
  backends.push_back(EncoderBackendType::kNvidiaNvEnc);
#endif
  return joinAvailable(backends);
}
