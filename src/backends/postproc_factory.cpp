#include "backend_registry.hpp"
#include "postproc_interface.hpp"
#include "backends/yolo_postproc.hpp"

#include <stdexcept>

PostprocBackendType detectAvailablePostprocBackend() {
  return PostprocBackendType::kYoloV8;
}

std::unique_ptr<IPostprocessor> createPostprocBackend(PostprocBackendType type) {
  if (type == PostprocBackendType::kAuto) {
    type = detectAvailablePostprocBackend();
  }

  switch (type) {
    case PostprocBackendType::kYoloV8:
      return std::make_unique<YoloPostprocessor>(YoloVersion::kYolov8);

    case PostprocBackendType::kYolo26:
      return std::make_unique<YoloPostprocessor>(YoloVersion::kYolo26);

    case PostprocBackendType::kYoloV5:
      return std::make_unique<YoloPostprocessor>(YoloVersion::kYolov8);

    default:
      throw std::runtime_error(
          "Postprocessor backend '" + toString(type) +
          "' is not available in this build. Available: " + availablePostprocBackends());
  }
}
