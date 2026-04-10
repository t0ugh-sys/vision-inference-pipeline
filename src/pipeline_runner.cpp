#include "pipeline_runner.hpp"

#include "backend_registry.hpp"
#include "decoder_interface.hpp"
#include "ffmpeg_packet_source.hpp"
#include "infer_interface.hpp"
#include "postproc_interface.hpp"
#include "preproc_interface.hpp"
#include "visualizer.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

template <typename BackendType>
void requireCompiledIn(
    BackendType type,
    const char* stageName,
    bool (*predicate)(BackendType),
    std::string (*availableFn)(),
    std::string (*nameFn)(BackendType)) {
  if (type == BackendType::kAuto || predicate(type)) {
    return;
  }

  throw std::runtime_error(
      std::string("Requested ") + stageName + " backend '" + nameFn(type) +
      "' is not available in this build. Available: " + availableFn());
}

}  // namespace

void validateAppConfig(const AppConfig& config) {
  if (config.source.uri.empty()) {
    throw std::runtime_error("Input source is required");
  }

  if (config.model.modelPath.empty()) {
    throw std::runtime_error("Model path is required");
  }

  if (config.model.inputWidth <= 0 || config.model.inputHeight <= 0) {
    throw std::runtime_error("Model input size must be positive");
  }

  if (config.maxFrames < 0) {
    throw std::runtime_error("maxFrames must be greater than or equal to 0");
  }

  requireCompiledIn(
      config.decoderBackend,
      "decoder",
      isCompiledIn,
      availableDecoderBackends,
      toString);
  requireCompiledIn(
      config.preprocBackend,
      "preprocessor",
      isCompiledIn,
      availablePreprocBackends,
      toString);
  requireCompiledIn(
      config.inferBackend,
      "inference",
      isCompiledIn,
      availableInferBackends,
      toString);
  requireCompiledIn(
      config.postprocBackend,
      "postprocessor",
      isCompiledIn,
      availablePostprocBackends,
      toString);

  const bool needsVisualization =
      config.visual.display ||
      !config.visual.outputVideo.empty() ||
      !config.visual.outputRtsp.empty();
  if (needsVisualization) {
    const auto visualizer = createVisualizer();
    if (!visualizer->isAvailable()) {
      throw std::runtime_error(
          "Visualization output requested, but no visualizer backend is available in this build");
    }
  }
}

void runPipeline(const AppConfig& config) {
  std::cout << "=== Video Inference Pipeline ===\n";
  std::cout << "Source: " << config.source.uri << "\n";
  std::cout << "Model: " << config.model.modelPath << "\n";
  std::cout << "Input: " << config.model.inputWidth << "x" << config.model.inputHeight << "\n";
  std::cout << "Max Frames: " << config.maxFrames << "\n";
  std::cout << "===============================\n\n";

  auto decoder = createDecoderBackend(config.decoderBackend);
  std::cout << "[1/6] Decoder: " << decoder->name() << "\n";

  auto preproc = createPreprocBackend(config.preprocBackend);
  std::cout << "[2/6] Preprocessor: " << preproc->name() << "\n";

  auto infer = createInferBackend(config.inferBackend);
  std::cout << "[3/6] Inference: " << infer->name() << "\n";
  infer->open(config.model);
  std::cout << "[4/6] Model loaded, input: " << infer->inputWidth() << "x" << infer->inputHeight() << "\n";

  auto postproc = createPostprocBackend(config.postprocBackend);
  std::cout << "[5/6] Postprocessor: " << postproc->name() << "\n";

  auto visualizer = createVisualizer();
  std::cout << "[6/6] Visualizer: " << visualizer->name() << "\n";

  FFmpegPacketSource packetSource;
  packetSource.open(config.source);
  decoder->open(packetSource.codec());

  bool visualizerInitialized = false;
  std::size_t frameCount = 0;

  while (true) {
    const EncodedPacket packet = packetSource.readPacket();
    const std::optional<DecodedFrame> decodedFrame = decoder->decode(packet);

    if (packet.endOfStream && !decodedFrame.has_value()) {
      break;
    }

    if (!decodedFrame.has_value()) {
      continue;
    }

    const RgbImage image = preproc->convertAndResize(
        decodedFrame.value(),
        infer->inputWidth(),
        infer->inputHeight());

    const std::vector<float> output = infer->infer(image);
    const DetectionResult result = postproc->postprocess(
        output,
        infer->inputWidth(),
        infer->inputHeight(),
        decodedFrame->width,
        decodedFrame->height,
        decodedFrame->pts);

    ++frameCount;
    std::cout << "frame=" << frameCount
              << " pts=" << decodedFrame->pts
              << " detections=" << result.boxes.size() << "\n";

    for (const auto& box : result.boxes) {
      std::cout << "  [" << box.label << " conf=" << box.score
                << " box=" << box.x1 << "," << box.y1
                << "-" << box.x2 << "," << box.y2 << "]\n";
    }

    if (config.visual.display || !config.visual.outputVideo.empty() || !config.visual.outputRtsp.empty()) {
      if (!visualizerInitialized && decodedFrame->width > 0 && decodedFrame->height > 0) {
        visualizer->init(decodedFrame->width, decodedFrame->height, config.visual);
        visualizerInitialized = true;
      }

      if (visualizerInitialized) {
        const RgbImage drawnImage = visualizer->draw(image, result);
        (void)drawnImage;
        visualizer->show();
      }
    }

    if (config.maxFrames > 0 && frameCount >= static_cast<std::size_t>(config.maxFrames)) {
      std::cout << "Reached max frames (" << config.maxFrames << "), stopping.\n";
      break;
    }
  }

  visualizer->close();

  std::cout << "\n=== Pipeline Complete ===\n";
  std::cout << "Total frames processed: " << frameCount << "\n";
}
