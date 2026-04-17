#include "pipeline_runner.hpp"

#include "backend_registry.hpp"
#include "decoder_interface.hpp"
#include "encoder_interface.hpp"
#include "ffmpeg_packet_source.hpp"
#include "infer_interface.hpp"
#include "postproc_interface.hpp"
#include "preproc_interface.hpp"
#include "visualizer.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct PreparedFrame {
  std::size_t index = 0;
  int64_t pts = 0;
  int originalWidth = 0;
  int originalHeight = 0;
  DecodedFrame decodedFrame;
  RgbImage inferenceImage;
  double decodeMs = 0.0;
  double preprocMs = 0.0;
};

struct ProcessedFrame {
  std::size_t index = 0;
  int64_t pts = 0;
  DecodedFrame decodedFrame;
  DetectionResult result;
  double decodeMs = 0.0;
  double preprocMs = 0.0;
  double inferMs = 0.0;
  double postprocMs = 0.0;
};

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  void push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    notFull_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
    if (closed_) {
      throw std::runtime_error("queue closed");
    }
    queue_.push_back(std::move(value));
    notEmpty_.notify_one();
  }

  bool pop(T& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    notEmpty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    value = std::move(queue_.front());
    queue_.pop_front();
    notFull_.notify_one();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    notEmpty_.notify_all();
    notFull_.notify_all();
  }

 private:
  std::size_t capacity_;
  std::deque<T> queue_;
  bool closed_ = false;
  std::mutex mutex_;
  std::condition_variable notEmpty_;
  std::condition_variable notFull_;
};

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

void maybeDumpFirstFrame(const AppConfig& config, const RgbImage& image, std::size_t frameCount) {
  if (!config.dumpFirstFrame || frameCount != 1) {
    return;
  }

  const std::string path = "dump_first_frame.ppm";
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open dump_first_frame.ppm for writing");
  }

  output << "P6\n" << image.width << " " << image.height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.data.data()), static_cast<std::streamsize>(image.data.size()));
}

PostprocessOptions makePostprocessOptions(const AppConfig& config) {
  return PostprocessOptions{
      config.confThreshold,
      config.nmsThreshold,
      config.labelsPath,
      {},
      config.modelOutputLayout,
      config.verbose};
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
  if (config.inferWorkers <= 0) {
    throw std::runtime_error("inferWorkers must be greater than 0");
  }

  requireCompiledIn(config.decoderBackend, "decoder", isCompiledIn, availableDecoderBackends, toString);
  requireCompiledIn(config.preprocBackend, "preprocessor", isCompiledIn, availablePreprocBackends, toString);
  requireCompiledIn(config.inferBackend, "inference", isCompiledIn, availableInferBackends, toString);
  requireCompiledIn(config.postprocBackend, "postprocessor", isCompiledIn, availablePostprocBackends, toString);

  if (!config.visual.outputVideo.empty() || !config.visual.outputRtsp.empty()) {
    throw std::runtime_error(
        "output-video/output-rtsp are disabled on the hardware-first path because they still depend on the OpenCV CPU visualizer. "
        "Use --display only, or implement a dedicated hardware encoder sink.");
  }

  if (config.visual.display) {
    const auto visualizer = createVisualizer();
    if (!visualizer->isAvailable()) {
      throw std::runtime_error(
          "Visualization display requested, but no visualizer backend is available in this build");
    }
  }
}

void runPipeline(const AppConfig& config) {
  const bool needsVisualization = config.visual.display;
  const bool needsDisplayFrame = needsVisualization || config.dumpFirstFrame;

  auto inferProbe = createInferBackend(config.inferBackend);
  inferProbe->open(config.model);
  const int inferInputWidth = inferProbe->inputWidth() > 0 ? inferProbe->inputWidth() : config.model.inputWidth;
  const int inferInputHeight = inferProbe->inputHeight() > 0 ? inferProbe->inputHeight() : config.model.inputHeight;

  auto decoder = createDecoderBackend(config.decoderBackend);
  auto preproc = createPreprocBackend(config.preprocBackend);

  FFmpegPacketSource packetSource;
  packetSource.open(config.source);
  decoder->open(packetSource.codec());

  BoundedQueue<PreparedFrame> preparedQueue(static_cast<std::size_t>(std::max(2, config.inferWorkers * 2)));
  BoundedQueue<ProcessedFrame> processedQueue(static_cast<std::size_t>(std::max(2, config.inferWorkers * 2)));

  std::exception_ptr workerError;
  std::mutex errorMutex;
  auto storeError = [&](std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(errorMutex);
    if (!workerError) {
      workerError = error;
    }
  };

  std::vector<std::thread> inferWorkers;
  inferWorkers.reserve(static_cast<std::size_t>(config.inferWorkers));
  for (int workerIndex = 0; workerIndex < config.inferWorkers; ++workerIndex) {
    inferWorkers.emplace_back([&, workerIndex] {
      try {
        auto infer = createInferBackend(config.inferBackend);
        infer->open(config.model);
        auto postproc = createPostprocBackend(config.postprocBackend, makePostprocessOptions(config));

        PreparedFrame prepared;
        while (preparedQueue.pop(prepared)) {
          const auto inferStart = Clock::now();
          const InferenceOutput output = infer->infer(prepared.inferenceImage);
          const auto inferEnd = Clock::now();
          const auto postStart = inferEnd;
          const DetectionResult result = postproc->postprocess(
              output,
              prepared.inferenceImage,
              prepared.originalWidth,
              prepared.originalHeight,
              prepared.pts);
          const auto postEnd = Clock::now();

          ProcessedFrame processed;
          processed.index = prepared.index;
          processed.pts = prepared.pts;
          processed.decodedFrame = std::move(prepared.decodedFrame);
          processed.result = result;
          processed.decodeMs = prepared.decodeMs;
          processed.preprocMs = prepared.preprocMs;
          processed.inferMs = Ms(inferEnd - inferStart).count();
          processed.postprocMs = Ms(postEnd - postStart).count();
          processedQueue.push(std::move(processed));
        }
      } catch (...) {
        storeError(std::current_exception());
        preparedQueue.close();
        processedQueue.close();
      }
    });
  }

  std::thread outputThread([&] {
    try {
      std::unique_ptr<IVisualizer> visualizer;
      std::unique_ptr<IPreprocessorBackend> displayPreproc;
      std::unique_ptr<IEncoderBackend> encoder;
      bool visualizerInitialized = false;
      bool encoderInitialized = false;
      if (needsVisualization) {
        visualizer = createVisualizer();
      }
      if (needsDisplayFrame) {
        displayPreproc = createPreprocBackend(config.preprocBackend);
      }
      if (!config.encoderOutput.empty()) {
        encoder = createEncoderBackend(EncoderBackendType::kAuto);
      }

      std::map<std::size_t, ProcessedFrame> pending;
      std::size_t nextIndex = 0;
      std::size_t displayedCount = 0;
      ProcessedFrame processed;
      while (processedQueue.pop(processed)) {
        pending.emplace(processed.index, std::move(processed));
        while (true) {
          auto it = pending.find(nextIndex);
          if (it == pending.end()) {
            break;
          }

          ProcessedFrame current = std::move(it->second);
          pending.erase(it);
          ++displayedCount;

          if (encoder && !encoderInitialized && current.decodedFrame.dmaFd >= 0) {
            EncoderConfig encCfg;
            encCfg.outputPath = config.encoderOutput;
            encCfg.codec = config.encoderCodec;
            encCfg.bitrate = config.encoderBitrate;
            encCfg.fps = config.encoderFps;
            encCfg.width = current.decodedFrame.width;
            encCfg.height = current.decodedFrame.height;
            encCfg.horStride = current.decodedFrame.horizontalStride > 0
                ? current.decodedFrame.horizontalStride
                : current.decodedFrame.width;
            encCfg.verStride = current.decodedFrame.verticalStride > 0
                ? current.decodedFrame.verticalStride
                : current.decodedFrame.height;
            encCfg.inputFormat = PixelFormat::kNv12;
            encoder->init(encCfg);
            encoderInitialized = true;
          }
          if (encoder && encoderInitialized && current.decodedFrame.dmaFd >= 0) {
            encoder->encodeDecodedFrame(current.decodedFrame, current.pts);
          }

          double displayPreprocMs = 0.0;
          std::optional<RgbImage> displayImage;
          if (needsDisplayFrame) {
            const auto displayPreprocStart = Clock::now();
            displayImage = displayPreproc->convertAndResize(
                current.decodedFrame,
                current.decodedFrame.width,
                current.decodedFrame.height,
                PreprocessOptions{});
            displayPreprocMs = Ms(Clock::now() - displayPreprocStart).count();
          }

          std::cout << "frame=" << displayedCount
                    << " pts=" << current.pts
                    << " detections=" << current.result.boxes.size();
          if (config.verbose) {
            std::cout << " decode_ms=" << current.decodeMs
                      << " preproc_ms=" << current.preprocMs
                      << " infer_ms=" << current.inferMs
                      << " post_ms=" << current.postprocMs;
            if (needsDisplayFrame) {
              std::cout << " display_preproc_ms=" << displayPreprocMs;
            }
          }
          std::cout << "\n";

          if (displayImage.has_value()) {
            maybeDumpFirstFrame(config, displayImage.value(), displayedCount);
          }

          if (needsVisualization && displayImage.has_value()) {
            if (!visualizerInitialized) {
              visualizer->init(displayImage->width, displayImage->height, config.visual);
              visualizerInitialized = true;
            }
            const RgbImage drawnImage = visualizer->draw(displayImage.value(), current.result);
            (void)drawnImage;
            visualizer->show();
          }

          ++nextIndex;
        }
      }

      if (encoder && encoderInitialized) {
        encoder->flush();
      }
      if (visualizer) {
        visualizer->close();
      }
    } catch (...) {
      storeError(std::current_exception());
      preparedQueue.close();
      processedQueue.close();
    }
  });

  try {
    bool eosSubmitted = false;
    std::size_t producedFrames = 0;
    while (!eosSubmitted && (config.maxFrames == 0 || producedFrames < static_cast<std::size_t>(config.maxFrames))) {
      const EncodedPacket packet = packetSource.readPacket();
      decoder->submitPacket(packet);
      eosSubmitted = packet.endOfStream;

      while (true) {
        const auto decodeStart = Clock::now();
        std::optional<DecodedFrame> decodedFrame = decoder->receiveFrame();
        const auto decodeEnd = Clock::now();
        if (!decodedFrame.has_value()) {
          break;
        }

        PreparedFrame prepared;
        prepared.index = producedFrames;
        prepared.pts = decodedFrame->pts;
        prepared.originalWidth = decodedFrame->width;
        prepared.originalHeight = decodedFrame->height;
        prepared.decodeMs = Ms(decodeEnd - decodeStart).count();

        const auto preprocStart = Clock::now();
        prepared.inferenceImage = preproc->convertAndResize(
            decodedFrame.value(),
            inferInputWidth,
            inferInputHeight,
            PreprocessOptions{config.letterbox, 114});
        prepared.decodedFrame = std::move(decodedFrame.value());
        const auto preprocEnd = Clock::now();
        prepared.preprocMs = Ms(preprocEnd - preprocStart).count();

        preparedQueue.push(std::move(prepared));
        ++producedFrames;
        if (config.maxFrames > 0 && producedFrames >= static_cast<std::size_t>(config.maxFrames)) {
          break;
        }
      }
    }
  } catch (...) {
    storeError(std::current_exception());
  }

  preparedQueue.close();
  for (auto& worker : inferWorkers) {
    worker.join();
  }
  processedQueue.close();
  outputThread.join();

  if (workerError) {
    std::rethrow_exception(workerError);
  }
}
