#include "pipeline_runner.hpp"

#include "backend_registry.hpp"
#include "decoder_interface.hpp"
#include "encoder_timing.hpp"
#include "encoder_interface.hpp"
#include "ffmpeg_packet_source.hpp"
#include "infer_interface.hpp"
#include "postproc_interface.hpp"
#include "preproc_interface.hpp"
#include "rga_shared.hpp"
#include "visualizer.hpp"

#include "../../rknn_model_zoo/utils/font.h"

#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
extern "C" {
#include <mpp_buffer.h>
}
#include <im2d.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

constexpr int kModelZooBoxThickness = 3;
constexpr int kModelZooFontPixelSize = 10;

struct RgbColor {
  std::uint8_t r = 0;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
};

struct YuvColor {
  std::uint8_t y = 0;
  std::uint8_t u = 128;
  std::uint8_t v = 128;
};

constexpr RgbColor kUltralyticsPalette[] = {
    {4, 42, 255},   {11, 219, 235}, {243, 243, 243}, {0, 223, 183},  {17, 31, 104},
    {255, 111, 221}, {255, 68, 79}, {204, 237, 0},   {0, 243, 68},   {189, 0, 255},
    {0, 180, 255},  {221, 0, 186}, {0, 255, 255},   {38, 192, 0},   {1, 255, 179},
    {125, 36, 255}, {123, 0, 104}, {255, 27, 108},  {252, 109, 47}, {162, 255, 11},
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

std::pair<int, int> computeDisplaySize(const AppConfig& config, int sourceWidth, int sourceHeight) {
  const int maxWidth = config.visual.displayMaxWidth;
  const int maxHeight = config.visual.displayMaxHeight;
  if (maxWidth <= 0 && maxHeight <= 0) {
    return {sourceWidth, sourceHeight};
  }

  float scale = 1.0f;
  if (maxWidth > 0) {
    scale = std::min(scale, static_cast<float>(maxWidth) / static_cast<float>(sourceWidth));
  }
  if (maxHeight > 0) {
    scale = std::min(scale, static_cast<float>(maxHeight) / static_cast<float>(sourceHeight));
  }
  scale = std::min(scale, 1.0f);

  const int width = std::max(1, static_cast<int>(sourceWidth * scale));
  const int height = std::max(1, static_cast<int>(sourceHeight * scale));
  return {width, height};
}

DetectionResult scaleDetectionResult(const DetectionResult& result, int targetWidth, int targetHeight) {
  if (result.imageWidth <= 0 || result.imageHeight <= 0 ||
      (result.imageWidth == targetWidth && result.imageHeight == targetHeight)) {
    return result;
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(result.imageWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(result.imageHeight);

  DetectionResult scaled = result;
  scaled.imageWidth = targetWidth;
  scaled.imageHeight = targetHeight;
  for (auto& box : scaled.boxes) {
    box.x1 *= scaleX;
    box.x2 *= scaleX;
    box.y1 *= scaleY;
    box.y2 *= scaleY;
  }
  return scaled;
}

RknnCoreMaskMode resolveAutoRknnCoreMask(int workerIndex, int workerCount) {
  switch (workerCount) {
    case 1:
      return RknnCoreMaskMode::kCore0_1_2;
    case 2:
      return workerIndex == 0 ? RknnCoreMaskMode::kCore0_1 : RknnCoreMaskMode::kCore2;
    case 3:
      if (workerIndex == 0) return RknnCoreMaskMode::kCore0;
      if (workerIndex == 1) return RknnCoreMaskMode::kCore1;
      return RknnCoreMaskMode::kCore2;
    default:
      return RknnCoreMaskMode::kAuto;
  }
}

InferRuntimeConfig makeInferRuntimeConfig(const AppConfig& config, int workerIndex, int workerCount) {
  InferRuntimeConfig runtime;
  runtime.workerIndex = workerIndex;
  runtime.workerCount = std::max(1, workerCount);
  runtime.verbose = config.verbose;
  runtime.rknnCoreMask =
      config.rknnCoreMask == RknnCoreMaskMode::kAuto
          ? resolveAutoRknnCoreMask(workerIndex, runtime.workerCount)
          : config.rknnCoreMask;
  return runtime;
}

int resolveInferWorkerCount(const AppConfig& config, InferBackendType selectedInferBackend) {
  if (config.inferWorkers > 0) {
    return config.inferWorkers;
  }
  if (selectedInferBackend == InferBackendType::kRockchipRknn) {
    return 3;
  }
  return 1;
}

std::string toLowerAscii(std::string value) {
  for (char& ch : value) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return value;
}

bool hasSuffixIgnoreCase(const std::string& value, const std::string& suffix) {
  const std::string lowerValue = toLowerAscii(value);
  const std::string lowerSuffix = toLowerAscii(suffix);
  return lowerValue.size() >= lowerSuffix.size() &&
         lowerValue.compare(lowerValue.size() - lowerSuffix.size(), lowerSuffix.size(), lowerSuffix) == 0;
}

bool startsWithIgnoreCase(const std::string& value, const std::string& prefix) {
  const std::string lowerValue = toLowerAscii(value);
  const std::string lowerPrefix = toLowerAscii(prefix);
  return lowerValue.size() >= lowerPrefix.size() &&
         lowerValue.compare(0, lowerPrefix.size(), lowerPrefix) == 0;
}

bool isRtspUrl(const std::string& value) {
  return startsWithIgnoreCase(value, "rtsp://");
}

std::string annotatedOutputTarget(const AppConfig& config) {
  if (!config.visual.outputRtsp.empty()) {
    return config.visual.outputRtsp;
  }
  return config.visual.outputVideo;
}

bool hasAnnotatedOutputTarget(const AppConfig& config) {
  return !annotatedOutputTarget(config).empty();
}

PostprocBackendType resolvePostprocBackend(const AppConfig& config) {
  if (config.postprocBackend != PostprocBackendType::kAuto) {
    return config.postprocBackend;
  }

  if (config.modelOutputLayout == ModelOutputLayout::kYolo26E2E) {
    return PostprocBackendType::kYolo26;
  }

  return PostprocBackendType::kYoloV8;
}

const char* toModelOutputLayoutName(ModelOutputLayout layout) {
  switch (layout) {
    case ModelOutputLayout::kAuto:
      return "auto";
    case ModelOutputLayout::kYolov8Flat:
      return "yolov8_flat_8400x84";
    case ModelOutputLayout::kYolov8RknnBranch6:
      return "yolov8_rknn_branch_6";
    case ModelOutputLayout::kYolov8RknnBranch9:
      return "yolov8_rknn_branch_9";
    case ModelOutputLayout::kYolo26E2E:
      return "yolo26_e2e";
  }
  return "unknown";
}

bool wantsHardwareEncodedAnnotatedOutput(const AppConfig& config) {
  const std::string outputTarget = annotatedOutputTarget(config);
  if (outputTarget.empty()) {
    return false;
  }

  if (isRtspUrl(outputTarget)) {
    return true;
  }

  return hasSuffixIgnoreCase(outputTarget, ".h264") ||
         hasSuffixIgnoreCase(outputTarget, ".264") ||
         hasSuffixIgnoreCase(outputTarget, ".h265") ||
         hasSuffixIgnoreCase(outputTarget, ".hevc") ||
         hasSuffixIgnoreCase(outputTarget, ".mp4");
}

OutputOverlayMode effectiveOutputOverlayMode(
    const AppConfig& config,
    EncoderBackendType selectedEncoderBackend) {
  if (config.outputOverlayExplicit) {
    return config.visual.outputOverlayMode;
  }

  if (hasAnnotatedOutputTarget(config) &&
      selectedEncoderBackend == EncoderBackendType::kRockchipMpp) {
    return OutputOverlayMode::kRga;
  }

  return config.visual.outputOverlayMode;
}

bool isRockchipRawHevcPath(const std::string& value) {
  return hasSuffixIgnoreCase(value, ".h265") || hasSuffixIgnoreCase(value, ".hevc");
}

bool usesRgaAnnotatedOutput(const AppConfig& config) {
  return wantsHardwareEncodedAnnotatedOutput(config);
}

const char* visualStyleName(VisualStyle style) {
  switch (style) {
    case VisualStyle::kClassic:
      return "classic";
    case VisualStyle::kYolo:
      return "yolo";
  }
  return "unknown";
}

const char* outputOverlayModeName(OutputOverlayMode mode) {
  switch (mode) {
    case OutputOverlayMode::kCpu:
      return "cpu";
    case OutputOverlayMode::kRga:
      return "rga";
  }
  return "unknown";
}

int computeAutoEncoderBitrate(int width, int height, int fps) {
  const long long pixelsPerSecond =
      static_cast<long long>(std::max(1, width)) *
      static_cast<long long>(std::max(1, height)) *
      static_cast<long long>(std::max(1, fps));
  const long long estimated = pixelsPerSecond / 10;
  const long long clamped = std::clamp<long long>(estimated, 4'000'000LL, 40'000'000LL);
  return static_cast<int>(clamped);
}

int resolveEncoderBitrate(const AppConfig& config, int width, int height, int fps) {
  if (config.encoderBitrate > 0) {
    return config.encoderBitrate;
  }
  return computeAutoEncoderBitrate(width, height, fps);
}

void fillRgbaRect(
    std::vector<std::uint8_t>& rgba,
    int imageWidth,
    int imageHeight,
    int x,
    int y,
    int width,
    int height,
    std::uint8_t r,
    std::uint8_t g,
    std::uint8_t b,
    std::uint8_t a) {
  const int x0 = std::clamp(x, 0, imageWidth);
  const int y0 = std::clamp(y, 0, imageHeight);
  const int x1 = std::clamp(x + width, 0, imageWidth);
  const int y1 = std::clamp(y + height, 0, imageHeight);
  if (x0 >= x1 || y0 >= y1) {
    return;
  }

  for (int yy = y0; yy < y1; ++yy) {
    for (int xx = x0; xx < x1; ++xx) {
      const std::size_t offset =
          static_cast<std::size_t>((yy * imageWidth + xx) * 4);
      rgba[offset + 0] = r;
      rgba[offset + 1] = g;
      rgba[offset + 2] = b;
      rgba[offset + 3] = a;
    }
  }
}

std::uint8_t clampToByte(int value) {
  return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

YuvColor rgbToNv12Color(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  const int y = ((66 * static_cast<int>(r) + 129 * static_cast<int>(g) + 25 * static_cast<int>(b) + 128) >> 8) + 16;
  const int u = ((-38 * static_cast<int>(r) - 74 * static_cast<int>(g) + 112 * static_cast<int>(b) + 128) >> 8) + 128;
  const int v = ((112 * static_cast<int>(r) - 94 * static_cast<int>(g) - 18 * static_cast<int>(b) + 128) >> 8) + 128;
  return {clampToByte(y), clampToByte(u), clampToByte(v)};
}

std::uint8_t alphaBlendByte(std::uint8_t dst, std::uint8_t src, std::uint8_t alpha) {
  const int invAlpha = 255 - static_cast<int>(alpha);
  return static_cast<std::uint8_t>(
      (static_cast<int>(dst) * invAlpha + static_cast<int>(src) * static_cast<int>(alpha) + 127) / 255);
}

void fillNv12Rect(
    std::uint8_t* yPlane,
    std::uint8_t* uvPlane,
    int yStride,
    int uvStride,
    int imageWidth,
    int imageHeight,
    int x,
    int y,
    int width,
    int height,
    const YuvColor& color,
    std::uint8_t alpha) {
  const int x0 = std::clamp(x, 0, imageWidth);
  const int y0 = std::clamp(y, 0, imageHeight);
  const int x1 = std::clamp(x + width, 0, imageWidth);
  const int y1 = std::clamp(y + height, 0, imageHeight);
  if (x0 >= x1 || y0 >= y1) {
    return;
  }

  if (alpha == 255) {
    for (int yy = y0; yy < y1; ++yy) {
      std::uint8_t* row = yPlane + static_cast<std::size_t>(yy) * yStride;
      std::memset(row + x0, color.y, static_cast<std::size_t>(x1 - x0));
    }

    const int uvX0 = x0 & ~1;
    const int uvY0 = y0 & ~1;
    const int uvX1 = (x1 + 1) & ~1;
    const int uvY1 = (y1 + 1) & ~1;
    for (int yy = uvY0; yy < uvY1; yy += 2) {
      std::uint8_t* row = uvPlane + static_cast<std::size_t>(yy / 2) * uvStride;
      for (int xx = uvX0; xx < uvX1; xx += 2) {
        row[xx + 0] = color.u;
        row[xx + 1] = color.v;
      }
    }
    return;
  }

  for (int yy = y0; yy < y1; ++yy) {
    std::uint8_t* row = yPlane + static_cast<std::size_t>(yy) * yStride;
    for (int xx = x0; xx < x1; ++xx) {
      row[xx] = alphaBlendByte(row[xx], color.y, alpha);
    }
  }

  const int uvX0 = x0 & ~1;
  const int uvY0 = y0 & ~1;
  const int uvX1 = (x1 + 1) & ~1;
  const int uvY1 = (y1 + 1) & ~1;
  for (int yy = uvY0; yy < uvY1; yy += 2) {
    std::uint8_t* row = uvPlane + static_cast<std::size_t>(yy / 2) * uvStride;
    for (int xx = uvX0; xx < uvX1; xx += 2) {
      row[xx + 0] = alphaBlendByte(row[xx + 0], color.u, alpha);
      row[xx + 1] = alphaBlendByte(row[xx + 1], color.v, alpha);
    }
  }
}

void drawNv12Rectangle(
    std::uint8_t* yPlane,
    std::uint8_t* uvPlane,
    int yStride,
    int uvStride,
    int imageWidth,
    int imageHeight,
    int x,
    int y,
    int width,
    int height,
    int thickness,
    const YuvColor& color,
    std::uint8_t alpha) {
  if (width <= 0 || height <= 0 || thickness <= 0) {
    return;
  }

  fillNv12Rect(yPlane, uvPlane, yStride, uvStride, imageWidth, imageHeight, x, y, width, thickness, color, alpha);
  fillNv12Rect(
      yPlane, uvPlane, yStride, uvStride, imageWidth, imageHeight, x, y + height - thickness, width, thickness, color, alpha);
  fillNv12Rect(yPlane, uvPlane, yStride, uvStride, imageWidth, imageHeight, x, y, thickness, height, color, alpha);
  fillNv12Rect(
      yPlane, uvPlane, yStride, uvStride, imageWidth, imageHeight, x + width - thickness, y, thickness, height, color, alpha);
}

void blendNv12Pixel(
    std::uint8_t* yPlane,
    std::uint8_t* uvPlane,
    int yStride,
    int uvStride,
    int imageWidth,
    int imageHeight,
    int x,
    int y,
    const YuvColor& color,
    std::uint8_t alpha) {
  if (x < 0 || y < 0 || x >= imageWidth || y >= imageHeight) {
    return;
  }

  if (alpha == 255) {
    yPlane[static_cast<std::size_t>(y) * yStride + x] = color.y;
    const int uvX = x & ~1;
    const int uvY = y & ~1;
    std::uint8_t* uv = uvPlane + static_cast<std::size_t>(uvY / 2) * uvStride + uvX;
    uv[0] = color.u;
    uv[1] = color.v;
    return;
  }

  yPlane[static_cast<std::size_t>(y) * yStride + x] =
      alphaBlendByte(yPlane[static_cast<std::size_t>(y) * yStride + x], color.y, alpha);

  const int uvX = x & ~1;
  const int uvY = y & ~1;
  std::uint8_t* uv = uvPlane + static_cast<std::size_t>(uvY / 2) * uvStride + uvX;
  uv[0] = alphaBlendByte(uv[0], color.u, alpha);
  uv[1] = alphaBlendByte(uv[1], color.v, alpha);
}

RgbColor ultralyticsColorForClass(int classId) {
  const std::size_t paletteSize = sizeof(kUltralyticsPalette) / sizeof(kUltralyticsPalette[0]);
  const std::size_t index =
      static_cast<std::size_t>(classId >= 0 ? classId : 0) % paletteSize;
  return kUltralyticsPalette[index];
}

RgbColor ultralyticsTextColor(const RgbColor& background) {
  const int luminance = static_cast<int>(background.r) + static_cast<int>(background.g) + static_cast<int>(background.b);
  if (luminance >= 600) {
    return {16, 16, 16};
  }
  return {255, 255, 255};
}

int resizeNearestC1(
    const unsigned char* srcPixels,
    int srcWidth,
    int srcHeight,
    unsigned char* dstPixels,
    int dstWidth,
    int dstHeight) {
  for (int i = 0; i < dstHeight; ++i) {
    const int y = std::clamp((i * srcHeight) / dstHeight, 0, srcHeight - 1);
    for (int j = 0; j < dstWidth; ++j) {
      const int x = std::clamp((j * srcWidth) / dstWidth, 0, srcWidth - 1);
      dstPixels[i * dstWidth + j] = srcPixels[y * srcWidth + x];
    }
  }

  return 0;
}

const std::vector<unsigned char>& resizedMonoGlyph(int fontBitmapIndex, int fontPixelSize) {
  static std::mutex cacheMutex;
  static std::map<int, std::vector<std::vector<unsigned char>>> cacheBySize;

  std::lock_guard<std::mutex> lock(cacheMutex);
  auto& cache = cacheBySize[fontPixelSize];
  if (cache.empty()) {
    cache.resize(95);
  }
  auto& glyph = cache[static_cast<std::size_t>(fontBitmapIndex)];
  if (glyph.empty()) {
    glyph.resize(static_cast<std::size_t>(fontPixelSize * fontPixelSize * 2));
    resizeNearestC1(
        mono_font_data[fontBitmapIndex],
        20,
        40,
        glyph.data(),
        fontPixelSize,
        fontPixelSize * 2);
  }
  return glyph;
}

void drawRectangleOnOverlay(
    std::vector<std::uint8_t>& rgba,
    int imageWidth,
    int imageHeight,
    int x,
    int y,
    int width,
    int height,
    int thickness,
    std::uint8_t r,
    std::uint8_t g,
    std::uint8_t b,
    std::uint8_t a) {
  if (width <= 0 || height <= 0 || thickness <= 0) {
    return;
  }

  fillRgbaRect(rgba, imageWidth, imageHeight, x, y, width, thickness, r, g, b, a);
  fillRgbaRect(rgba, imageWidth, imageHeight, x, y + height - thickness, width, thickness, r, g, b, a);
  fillRgbaRect(rgba, imageWidth, imageHeight, x, y, thickness, height, r, g, b, a);
  fillRgbaRect(rgba, imageWidth, imageHeight, x + width - thickness, y, thickness, height, r, g, b, a);
}

void drawTextOnOverlay(
    std::vector<std::uint8_t>& rgba,
    int width,
    int height,
    const char* text,
    int x,
    int y,
    int fontPixelSize,
    std::uint8_t r,
    std::uint8_t g,
    std::uint8_t b) {
  const int n = static_cast<int>(std::strlen(text));
  int cursorX = x;
  int cursorY = y;
  for (int i = 0; i < n; ++i) {
    const char ch = text[i];
    if (ch == '\n') {
      cursorX = x;
      cursorY += fontPixelSize * 2;
      continue;
    }
    if (std::isprint(static_cast<unsigned char>(ch)) == 0) {
      continue;
    }

    const int fontBitmapIndex = ch - ' ';
    if (fontBitmapIndex < 0 || fontBitmapIndex >= 95) {
      continue;
    }
    const auto& resizedFontBitmap = resizedMonoGlyph(fontBitmapIndex, fontPixelSize);

    for (int yy = cursorY; yy < cursorY + fontPixelSize * 2; ++yy) {
      if (yy < 0) {
        continue;
      }
      if (yy >= height) {
        break;
      }

      const unsigned char* alpha = resizedFontBitmap.data() +
          static_cast<std::size_t>(yy - cursorY) * fontPixelSize;
      for (int xx = cursorX; xx < cursorX + fontPixelSize; ++xx) {
        if (xx < 0) {
          continue;
        }
        if (xx >= width) {
          break;
        }

        const unsigned char a = alpha[xx - cursorX] >= 128 ? 255 : 0;
        const std::size_t offset =
            static_cast<std::size_t>((yy * width + xx) * 4);
        rgba[offset + 0] = static_cast<unsigned char>((rgba[offset + 0] * (255 - a) + r * a) / 255);
        rgba[offset + 1] = static_cast<unsigned char>((rgba[offset + 1] * (255 - a) + g * a) / 255);
        rgba[offset + 2] = static_cast<unsigned char>((rgba[offset + 2] * (255 - a) + b * a) / 255);
        rgba[offset + 3] = std::max(rgba[offset + 3], a);
      }
    }

    cursorX += fontPixelSize;
  }
}

void drawTextOnNv12(
    std::uint8_t* yPlane,
    std::uint8_t* uvPlane,
    int yStride,
    int uvStride,
    int imageWidth,
    int imageHeight,
    const char* text,
    int x,
    int y,
    int fontPixelSize,
    const YuvColor& color) {
  const int n = static_cast<int>(std::strlen(text));
  int cursorX = x;
  int cursorY = y;
  for (int i = 0; i < n; ++i) {
    const char ch = text[i];
    if (ch == '\n') {
      cursorX = x;
      cursorY += fontPixelSize * 2;
      continue;
    }
    if (std::isprint(static_cast<unsigned char>(ch)) == 0) {
      continue;
    }

    const int fontBitmapIndex = ch - ' ';
    if (fontBitmapIndex < 0 || fontBitmapIndex >= 95) {
      continue;
    }
    const auto& resizedFontBitmap = resizedMonoGlyph(fontBitmapIndex, fontPixelSize);

    for (int yy = 0; yy < fontPixelSize * 2; ++yy) {
      const int pixelY = cursorY + yy;
      if (pixelY < 0 || pixelY >= imageHeight) {
        continue;
      }

      const unsigned char* alphaRow =
          resizedFontBitmap.data() + static_cast<std::size_t>(yy) * fontPixelSize;
      for (int xx = 0; xx < fontPixelSize; ++xx) {
        const int pixelX = cursorX + xx;
        if (pixelX < 0 || pixelX >= imageWidth) {
          continue;
        }
        if (alphaRow[xx] < 128) {
          continue;
        }
        blendNv12Pixel(yPlane, uvPlane, yStride, uvStride, imageWidth, imageHeight, pixelX, pixelY, color, 255);
      }
    }

    cursorX += fontPixelSize;
  }
}

void drawYoloLabelBoxOnOverlay(
    std::vector<std::uint8_t>& rgba,
    int imageWidth,
    int imageHeight,
    const char* text,
    int anchorX,
    int anchorY,
    int fontPixelSize,
    std::uint8_t bgR,
    std::uint8_t bgG,
    std::uint8_t bgB,
    std::uint8_t bgA,
    std::uint8_t textR,
    std::uint8_t textG,
    std::uint8_t textB) {
  if (text == nullptr || text[0] == '\0') {
    return;
  }

  const int textLength = static_cast<int>(std::strlen(text));
  const int charWidth = std::max(6, fontPixelSize);
  const int padX = std::max(4, fontPixelSize / 2);
  const int padY = std::max(3, fontPixelSize / 3);
  const int labelWidth = textLength * charWidth + padX * 2;
  const int labelHeight = fontPixelSize * 2 + padY * 2;

  int labelX = std::clamp(anchorX, 0, std::max(0, imageWidth - 1));
  int labelY = anchorY - labelHeight;
  if (labelY < 0) {
    labelY = std::clamp(anchorY + 1, 0, std::max(0, imageHeight - 1));
  }

  fillRgbaRect(
      rgba,
      imageWidth,
      imageHeight,
      labelX,
      labelY,
      std::min(labelWidth, std::max(0, imageWidth - labelX)),
      std::min(labelHeight, std::max(0, imageHeight - labelY)),
      bgR,
      bgG,
      bgB,
      bgA);
  drawTextOnOverlay(
      rgba,
      imageWidth,
      imageHeight,
      text,
      labelX + padX,
      labelY + padY,
      fontPixelSize,
      textR,
      textG,
      textB);
}

#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
std::shared_ptr<void> makeMppBufferOwner(MppBuffer buffer) {
  return std::shared_ptr<void>(buffer, [](void* opaque) {
    if (opaque != nullptr) {
      mpp_buffer_put(static_cast<MppBuffer>(opaque));
    }
  });
}

DecodedFrame makeAnnotatedEncodeFrame(
    const DecodedFrame& frame,
    const DetectionResult& result,
    const VisualConfig& config) {
  if (frame.dmaFd < 0 || result.boxes.empty()) {
    return frame;
  }

  static MppBufferGroup outputGroup = nullptr;
  if (outputGroup == nullptr) {
    if (mpp_buffer_group_get_internal(
            &outputGroup,
            static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)) != MPP_OK) {
      throw std::runtime_error("mpp_buffer_group_get_internal failed for hardware overlay output");
    }
  }

  const int horStride = frame.horizontalStride > 0 ? frame.horizontalStride : frame.width;
  const int verStride = frame.verticalStride > 0 ? frame.verticalStride : frame.height;
  const int uvStride = frame.chromaStride > 0 ? frame.chromaStride : horStride;
  const std::size_t yBytes = static_cast<std::size_t>(horStride) * static_cast<std::size_t>(verStride);
  const std::size_t uvBytes = static_cast<std::size_t>(uvStride) * static_cast<std::size_t>(verStride / 2);
  const std::size_t outputBytes = yBytes + uvBytes;

  MppBuffer outputBuffer = nullptr;
  if (mpp_buffer_get(outputGroup, &outputBuffer, outputBytes) != MPP_OK || outputBuffer == nullptr) {
    throw std::runtime_error("mpp_buffer_get failed for hardware overlay NV12 output buffer");
  }
  try {
    MppBuffer srcBuffer = static_cast<MppBuffer>(frame.nativeHandle.get());
    auto* dstPtr = static_cast<std::uint8_t*>(mpp_buffer_get_ptr(outputBuffer));
    auto* srcPtr = srcBuffer != nullptr ? static_cast<std::uint8_t*>(mpp_buffer_get_ptr(srcBuffer)) : nullptr;
    if (dstPtr == nullptr) {
      throw std::runtime_error("Failed to map destination NV12 buffer");
    }

    if (srcPtr != nullptr) {
      std::memcpy(dstPtr, srcPtr, outputBytes);
    } else if (frame.yData.size() >= yBytes && frame.uvData.size() >= uvBytes) {
      std::memcpy(dstPtr, frame.yData.data(), yBytes);
      std::memcpy(dstPtr + yBytes, frame.uvData.data(), uvBytes);
    } else {
      throw std::runtime_error("Failed to access source NV12 pixels for annotated output");
    }
    std::uint8_t* yPlane = dstPtr;
    std::uint8_t* uvPlane = dstPtr + yBytes;
    const int thickness = std::max(1, kModelZooBoxThickness);

    for (const auto& box : result.boxes) {
      const int x1 = std::clamp(static_cast<int>(box.x1), 0, frame.width - 1);
      const int y1 = std::clamp(static_cast<int>(box.y1), 0, frame.height - 1);
      const int x2 = std::clamp(static_cast<int>(box.x2), x1 + 1, frame.width);
      const int y2 = std::clamp(static_cast<int>(box.y2), y1 + 1, frame.height);
      const RgbColor classColor = ultralyticsColorForClass(box.classId);
      const RgbColor boxColor =
          config.style == VisualStyle::kYolo ? classColor : RgbColor{0, 0, 255};
      const YuvColor boxYuv = rgbToNv12Color(boxColor.r, boxColor.g, boxColor.b);

      drawNv12Rectangle(
          yPlane,
          uvPlane,
          horStride,
          uvStride,
          frame.width,
          frame.height,
          x1,
          y1,
          std::max(1, x2 - x1),
          std::max(1, y2 - y1),
          thickness,
          boxYuv,
          255);

      char text[256] = {};
      if (config.showLabel && !box.label.empty() && config.showConf) {
        std::snprintf(text, sizeof(text), "%s %.1f%%", box.label.c_str(), box.score * 100.0f);
      } else if (config.showLabel && !box.label.empty()) {
        std::snprintf(text, sizeof(text), "%s", box.label.c_str());
      } else if (config.showConf) {
        std::snprintf(text, sizeof(text), "%.1f%%", box.score * 100.0f);
      }
      if (text[0] == '\0') {
        continue;
      }

      if (config.style == VisualStyle::kYolo) {
        const RgbColor textColor = ultralyticsTextColor(classColor);
        const YuvColor bgYuv = rgbToNv12Color(classColor.r, classColor.g, classColor.b);
        const YuvColor textYuv = rgbToNv12Color(textColor.r, textColor.g, textColor.b);
        const int textLength = static_cast<int>(std::strlen(text));
        const int charWidth = std::max(6, kModelZooFontPixelSize);
        const int padX = std::max(4, kModelZooFontPixelSize / 2);
        const int padY = std::max(3, kModelZooFontPixelSize / 3);
        const int labelWidth = textLength * charWidth + padX * 2;
        const int labelHeight = kModelZooFontPixelSize * 2 + padY * 2;
        int labelX = std::clamp(x1, 0, std::max(0, frame.width - 1));
        int labelY = y1 - labelHeight;
        if (labelY < 0) {
          labelY = std::clamp(y1 + 1, 0, std::max(0, frame.height - 1));
        }
        fillNv12Rect(
            yPlane,
            uvPlane,
            horStride,
            uvStride,
            frame.width,
            frame.height,
            labelX,
            labelY,
            std::min(labelWidth, std::max(0, frame.width - labelX)),
            std::min(labelHeight, std::max(0, frame.height - labelY)),
            bgYuv,
            255);
        drawTextOnNv12(
            yPlane,
            uvPlane,
            horStride,
            uvStride,
            frame.width,
            frame.height,
            text,
            labelX + padX,
            labelY + padY,
            kModelZooFontPixelSize,
            textYuv);
      } else {
        drawTextOnNv12(
            yPlane,
            uvPlane,
            horStride,
            uvStride,
            frame.width,
            frame.height,
            text,
            x1,
            y1 - 20,
            kModelZooFontPixelSize,
            rgbToNv12Color(255, 0, 0));
      }
    }

    DecodedFrame annotated = frame;
    annotated.dmaFd = mpp_buffer_get_fd(outputBuffer);
    annotated.nativeHandle = makeMppBufferOwner(outputBuffer);
    return annotated;
  } catch (...) {
    if (outputBuffer != nullptr) mpp_buffer_put(outputBuffer);
    throw;
  }
}
#else
DecodedFrame makeAnnotatedEncodeFrame(const DecodedFrame&, const DetectionResult&, const VisualConfig&) {
  throw std::runtime_error("Hardware box drawing requires Rockchip RGA support");
}
#endif

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
  if (config.inferWorkers < 0) {
    throw std::runtime_error("inferWorkers must be greater than or equal to 0");
  }

  requireCompiledIn(config.decoderBackend, "decoder", isCompiledIn, availableDecoderBackends, toString);
  requireCompiledIn(config.preprocBackend, "preprocessor", isCompiledIn, availablePreprocBackends, toString);
  requireCompiledIn(config.inferBackend, "inference", isCompiledIn, availableInferBackends, toString);
  requireCompiledIn(
      resolvePostprocBackend(config),
      "postprocessor",
      isCompiledIn,
      availablePostprocBackends,
      toString);
  const EncoderBackendType selectedEncoderBackend = detectAvailableEncoderBackend();
  const OutputOverlayMode outputOverlayMode =
      effectiveOutputOverlayMode(config, selectedEncoderBackend);

  if (!config.visual.outputVideo.empty() && !config.visual.outputRtsp.empty()) {
    throw std::runtime_error("Specify only one annotated output target: use either --output-video or --output-rtsp");
  }

  if (!config.visual.outputRtsp.empty() && !isRtspUrl(config.visual.outputRtsp)) {
    throw std::runtime_error("output-rtsp must start with rtsp://");
  }

  if (hasAnnotatedOutputTarget(config) &&
      !wantsHardwareEncodedAnnotatedOutput(config) &&
      selectedEncoderBackend == EncoderBackendType::kRockchipMpp) {
    throw std::runtime_error(
        "On the Rockchip path, annotated output uses hardware encoding. "
        "Use .h264/.264 raw bitstream, .mp4, or rtsp:// for muxed streaming output.");
  }

  if (!config.encoderOutput.empty()) {
    auto encoder = createEncoderBackend(EncoderBackendType::kAuto);
    if (selectedEncoderBackend == EncoderBackendType::kRockchipMpp && config.encoderCodec == "h265") {
      throw std::runtime_error(
          "The Rockchip MPP encoder path does not support --encoder-codec h265 yet. Use h264.");
    }
    if (!encoder->supportsDecodedFrameInput()) {
      throw std::runtime_error(
          "encoder-output currently requires an encoder backend that accepts decoded NV12 frames. "
          "The selected auto encoder is '" +
          encoder->name() +
          "', which only accepts RGB images. Use --output-video for the NVIDIA path or switch to Rockchip MPP.");
    }
    if (selectedEncoderBackend == EncoderBackendType::kRockchipMpp &&
        isRockchipRawHevcPath(config.encoderOutput)) {
      throw std::runtime_error(
          "The Rockchip MPP encoder path does not support .h265/.hevc output yet. Use .h264 or .mp4.");
    }
  }

  if (outputOverlayMode == OutputOverlayMode::kRga &&
      hasAnnotatedOutputTarget(config) &&
      selectedEncoderBackend == EncoderBackendType::kNvidiaNvEnc) {
    throw std::runtime_error(
        "output-overlay=rga is only available on the Rockchip RGA path. Use --output-overlay cpu on the NVIDIA platform.");
  }

  if (hasAnnotatedOutputTarget(config) &&
      selectedEncoderBackend == EncoderBackendType::kRockchipMpp &&
      config.encoderCodec == "h265") {
    throw std::runtime_error(
        "The Rockchip annotated output path does not support --encoder-codec h265 yet. Use h264.");
  }

  if (!config.visual.outputVideo.empty() &&
      selectedEncoderBackend == EncoderBackendType::kRockchipMpp &&
      isRockchipRawHevcPath(config.visual.outputVideo)) {
    throw std::runtime_error(
        "The Rockchip annotated output path does not support .h265/.hevc output yet. Use .h264, .mp4, or --output-rtsp.");
  }

  if (config.visual.display ||
      (hasAnnotatedOutputTarget(config) && outputOverlayMode != OutputOverlayMode::kRga)) {
    const auto visualizer = createVisualizer();
    if (!visualizer->isAvailable()) {
      throw std::runtime_error(
          "Visualization requested, but no visualizer backend is available in this build");
    }
  }
}

void runPipeline(const AppConfig& config) {
  const std::string annotatedOutputPath = annotatedOutputTarget(config);
  const EncoderBackendType selectedEncoderBackend = detectAvailableEncoderBackend();
  const OutputOverlayMode outputOverlayMode =
      effectiveOutputOverlayMode(config, selectedEncoderBackend);
  const bool needsHardwareEncodedAnnotatedVideo = wantsHardwareEncodedAnnotatedOutput(config);
  const bool useRgaAnnotatedOutput =
      needsHardwareEncodedAnnotatedVideo && outputOverlayMode == OutputOverlayMode::kRga;
  const bool needsVisualizerDraw = config.visual.display ||
                                   (!annotatedOutputPath.empty() && !useRgaAnnotatedOutput);
  const bool needsDisplayFrame = needsVisualizerDraw || config.dumpFirstFrame;
  const PostprocBackendType resolvedPostprocBackend = resolvePostprocBackend(config);
  const DecoderBackendType selectedDecoderBackend =
      config.decoderBackend == DecoderBackendType::kAuto ? detectAvailableDecoderBackend() : config.decoderBackend;
  const PreprocBackendType selectedPreprocBackend =
      config.preprocBackend == PreprocBackendType::kAuto ? detectAvailablePreprocBackend() : config.preprocBackend;
  const InferBackendType selectedInferBackend =
      config.inferBackend == InferBackendType::kAuto ? detectAvailableInferBackend() : config.inferBackend;
  const int resolvedInferWorkers = resolveInferWorkerCount(config, selectedInferBackend);
  const std::size_t inferenceQueueCapacity = static_cast<std::size_t>(std::max(2, resolvedInferWorkers * 2));
  const std::size_t rgaMaxInflightFrames = static_cast<std::size_t>(std::max(1, resolvedInferWorkers * 2 + 4));

  if (config.verbose) {
    std::cerr << "[PIPELINE] capabilities"
              << " compiled_decoder=" << availableDecoderBackends()
              << " compiled_preproc=" << availablePreprocBackends()
              << " compiled_infer=" << availableInferBackends()
              << " compiled_postproc=" << availablePostprocBackends()
              << " compiled_encoder=" << availableEncoderBackends()
              << " selected_decoder=" << toString(selectedDecoderBackend)
              << " selected_preproc=" << toString(selectedPreprocBackend)
              << " selected_infer=" << toString(selectedInferBackend)
              << " selected_postproc=" << toString(resolvedPostprocBackend)
              << " selected_encoder=" << toString(selectedEncoderBackend)
              << " input_uri=" << config.source.uri
              << " annotated_output=" << (annotatedOutputPath.empty() ? "none" : annotatedOutputPath)
              << " encoder_output=" << (config.encoderOutput.empty() ? "none" : config.encoderOutput)
              << " overlay=" << outputOverlayModeName(outputOverlayMode)
              << " visual_style=" << visualStyleName(config.visual.style)
              << " letterbox=" << (config.letterbox ? "on" : "off")
              << " rknn_zero_copy=" << (config.rknnZeroCopy ? "on" : "off")
              << " model_output_layout=" << toModelOutputLayoutName(config.modelOutputLayout)
              << " infer_workers=" << resolvedInferWorkers
              << "\n";
    std::cerr << "[PIPELINE] postproc requested=" << toString(config.postprocBackend)
              << " resolved=" << toString(resolvedPostprocBackend)
              << " model_output_layout=" << toModelOutputLayoutName(config.modelOutputLayout)
              << " model=" << config.model.modelPath << "\n";
  }

  int inferInputWidth = config.model.inputWidth;
  int inferInputHeight = config.model.inputHeight;
  {
    auto inferProbe = createInferBackend(config.inferBackend);
    inferProbe->open(config.model, makeInferRuntimeConfig(config, 0, resolvedInferWorkers));
    inferInputWidth = inferProbe->inputWidth() > 0 ? inferProbe->inputWidth() : config.model.inputWidth;
    inferInputHeight = inferProbe->inputHeight() > 0 ? inferProbe->inputHeight() : config.model.inputHeight;
    if (config.verbose) {
      std::cerr << "[PIPELINE] infer_probe backend=" << inferProbe->name()
                << " input=" << inferInputWidth << "x" << inferInputHeight << "\n";
    }
  }

  auto decoder = createDecoderBackend(config.decoderBackend);
  auto preproc = createPreprocBackend(config.preprocBackend);
  preproc->setMaxInflightFrames(rgaMaxInflightFrames);

  FFmpegPacketSource packetSource;
  packetSource.open(config.source);
  const SourceVideoInfo sourceVideoInfo = packetSource.videoInfo();
  decoder->open(packetSource.codec());
  if (config.verbose) {
    if (!packetSource.inputOptionsSummary().empty()) {
      std::cerr << "[PIPELINE] input_options " << packetSource.inputOptionsSummary() << "\n";
    }
    std::cerr << "[PIPELINE] stages decoder=" << decoder->name()
              << " preproc=" << preproc->name()
              << " infer=" << toString(selectedInferBackend)
              << " postproc=" << toString(resolvedPostprocBackend)
              << " source=" << sourceVideoInfo.width << "x" << sourceVideoInfo.height
              << " fps=" << sourceVideoInfo.fpsNum << "/" << sourceVideoInfo.fpsDen
              << " infer_workers=" << resolvedInferWorkers << "\n";
  }

  BoundedQueue<PreparedFrame> preparedQueue(inferenceQueueCapacity);
  BoundedQueue<ProcessedFrame> processedQueue(inferenceQueueCapacity);

  std::exception_ptr workerError;
  std::mutex errorMutex;
  auto storeError = [&](std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(errorMutex);
    if (!workerError) {
      workerError = error;
    }
  };

  std::vector<std::thread> inferWorkers;
  inferWorkers.reserve(static_cast<std::size_t>(resolvedInferWorkers));
  for (int workerIndex = 0; workerIndex < resolvedInferWorkers; ++workerIndex) {
    inferWorkers.emplace_back([&, workerIndex] {
      try {
        auto infer = createInferBackend(config.inferBackend);
        infer->open(config.model, makeInferRuntimeConfig(config, workerIndex, resolvedInferWorkers));
        auto postproc = createPostprocBackend(resolvedPostprocBackend, makePostprocessOptions(config));

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
      std::unique_ptr<IEncoderBackend> annotatedVideoEncoder;
      bool visualizerInitialized = false;
      bool encoderInitialized = false;
      bool annotatedVideoEncoderInitialized = false;
      const int outputEncoderFps = resolveEncoderFps(config, sourceVideoInfo);
      if (needsVisualizerDraw) {
        visualizer = createVisualizer();
      }
      if (needsDisplayFrame) {
        displayPreproc = createPreprocBackend(config.preprocBackend);
        displayPreproc->setMaxInflightFrames(rgaMaxInflightFrames);
      }
      if (!config.encoderOutput.empty()) {
        encoder = createEncoderBackend(EncoderBackendType::kAuto);
        if (config.verbose) {
          std::cerr << "[PIPELINE] raw_encoder backend=" << encoder->name() << "\n";
        }
      }
      if (needsHardwareEncodedAnnotatedVideo) {
        annotatedVideoEncoder = createEncoderBackend(EncoderBackendType::kAuto);
        if (config.verbose) {
          std::cerr << "[PIPELINE] annotated_encoder backend=" << annotatedVideoEncoder->name() << "\n";
        }
      }

      std::map<std::size_t, ProcessedFrame> pending;
      std::size_t nextIndex = 0;
      std::size_t displayedCount = 0;
      const auto outputStart = Clock::now();
      double totalOverlayMs = 0.0;
      double totalAnnotatedEncodeMs = 0.0;
      std::size_t timedAnnotatedFrames = 0;
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

          if (encoder && current.decodedFrame.dmaFd < 0) {
            throw std::runtime_error(
                "encoder-output requested, but decoded frame does not provide a valid dma fd");
          }
          if (encoder && !encoderInitialized) {
            EncoderConfig encCfg;
            encCfg.outputPath = config.encoderOutput;
            encCfg.codec = config.encoderCodec;
            encCfg.fps = outputEncoderFps;
            encCfg.fpsNum = resolveEncoderFpsNum(config, sourceVideoInfo);
            encCfg.fpsDen = resolveEncoderFpsDen(config, sourceVideoInfo);
            encCfg.width = current.decodedFrame.width;
            encCfg.height = current.decodedFrame.height;
            encCfg.horStride = current.decodedFrame.horizontalStride > 0
                ? current.decodedFrame.horizontalStride
                : current.decodedFrame.width;
            encCfg.verStride = current.decodedFrame.verticalStride > 0
                ? current.decodedFrame.verticalStride
                : current.decodedFrame.height;
            encCfg.bitrate = resolveEncoderBitrate(config, encCfg.width, encCfg.height, outputEncoderFps);
            encCfg.inputFormat = PixelFormat::kNv12;
            encCfg.lowLatency = config.encoderLowLatency;
            if (config.verbose) {
              std::cerr << "[PIPELINE] init_raw_encoder backend=" << encoder->name()
                        << " codec=" << encCfg.codec
                        << " path=" << encCfg.outputPath
                        << " size=" << encCfg.width << "x" << encCfg.height
                        << " stride=" << encCfg.horStride << "x" << encCfg.verStride
                        << " fps=" << encCfg.fpsNum << "/" << encCfg.fpsDen
                        << " bitrate=" << encCfg.bitrate << "\n";
            }
            encoder->init(encCfg);
            encoderInitialized = true;
          }
          const bool keepEncodedFrame =
              shouldKeepEncodedFrame(displayedCount - 1, sourceVideoInfo, outputEncoderFps);
          if (encoder && encoderInitialized && keepEncodedFrame) {
            encoder->encodeDecodedFrame(current.decodedFrame, current.pts);
          }

          double displayPreprocMs = 0.0;
          std::optional<RgbImage> displayImage;
          std::optional<DetectionResult> displayResult;
          if (needsDisplayFrame) {
            const auto [displayWidth, displayHeight] =
                needsVisualizerDraw
                    ? computeDisplaySize(config, current.decodedFrame.width, current.decodedFrame.height)
                    : std::pair<int, int>{current.decodedFrame.width, current.decodedFrame.height};
            const auto displayPreprocStart = Clock::now();
            displayImage = displayPreproc->convertAndResize(
                current.decodedFrame,
                displayWidth,
                displayHeight,
                PreprocessOptions{false, 114, true});
            displayResult = scaleDetectionResult(current.result, displayWidth, displayHeight);
            displayPreprocMs = Ms(Clock::now() - displayPreprocStart).count();
          }

          const bool shouldLogProgress =
              displayedCount == 1 ||
              (config.progressEvery > 0 && (displayedCount % static_cast<std::size_t>(config.progressEvery) == 0));
          if (shouldLogProgress) {
            const double elapsedSeconds =
                std::max(1e-6, std::chrono::duration<double>(Clock::now() - outputStart).count());
            const double fps = static_cast<double>(displayedCount) / elapsedSeconds;
            std::cout << "frame=" << displayedCount
                      << " pts=" << current.pts
                      << " detections=" << current.result.boxes.size()
                      << " fps=" << fps;
            if (config.verbose) {
              std::cout << " decode_ms=" << current.decodeMs
                        << " preproc_ms=" << current.preprocMs
                        << " infer_ms=" << current.inferMs
                        << " post_ms=" << current.postprocMs;
              if (needsDisplayFrame) {
                std::cout << " display_preproc_ms=" << displayPreprocMs;
              }
              if (timedAnnotatedFrames > 0) {
                std::cout << " overlay_ms_avg=" << (totalOverlayMs / timedAnnotatedFrames)
                          << " annot_enc_ms_avg=" << (totalAnnotatedEncodeMs / timedAnnotatedFrames);
              }
            }
            std::cout << "\n";
          }

          if (displayImage.has_value()) {
            maybeDumpFirstFrame(config, displayImage.value(), displayedCount);
          }

          if (needsVisualizerDraw && displayImage.has_value() && displayResult.has_value()) {
            if (!visualizerInitialized) {
              VisualConfig visualConfig = config.visual;
              if (needsHardwareEncodedAnnotatedVideo) {
                visualConfig.outputVideo.clear();
                visualConfig.outputRtsp.clear();
              }
              if (config.verbose) {
                std::cerr << "[PIPELINE] init_visualizer size="
                          << displayImage->width << "x" << displayImage->height << "\n";
              }
              visualizer->init(displayImage->width, displayImage->height, visualConfig);
              visualizerInitialized = true;
            }
            const RgbImage drawnImage = visualizer->draw(displayImage.value(), displayResult.value());
            if (annotatedVideoEncoder) {
              if (!annotatedVideoEncoderInitialized) {
                EncoderConfig encCfg;
                encCfg.outputPath = annotatedOutputPath;
                encCfg.codec = config.encoderCodec;
                encCfg.fps = outputEncoderFps;
                encCfg.fpsNum = resolveEncoderFpsNum(config, sourceVideoInfo);
                encCfg.fpsDen = resolveEncoderFpsDen(config, sourceVideoInfo);
                encCfg.width = useRgaAnnotatedOutput ? current.decodedFrame.width : drawnImage.width;
                encCfg.height = useRgaAnnotatedOutput ? current.decodedFrame.height : drawnImage.height;
                encCfg.bitrate = resolveEncoderBitrate(config, encCfg.width, encCfg.height, outputEncoderFps);
                encCfg.inputFormat = useRgaAnnotatedOutput ? PixelFormat::kNv12 : PixelFormat::kRgb888;
                encCfg.lowLatency = config.encoderLowLatency;
                if (config.verbose) {
                  std::cerr << "[PIPELINE] init_annotated_encoder backend=" << annotatedVideoEncoder->name()
                            << " codec=" << encCfg.codec
                            << " path=" << encCfg.outputPath
                            << " size=" << encCfg.width << "x" << encCfg.height
                            << " fps=" << encCfg.fpsNum << "/" << encCfg.fpsDen
                            << " bitrate=" << encCfg.bitrate
                            << " input_format=" << (encCfg.inputFormat == PixelFormat::kNv12 ? "NV12" : "RGB888")
                            << "\n";
                }
                annotatedVideoEncoder->init(encCfg);
                annotatedVideoEncoderInitialized = true;
              }
              if (keepEncodedFrame) {
                if (useRgaAnnotatedOutput) {
                  const auto overlayStart = Clock::now();
                  DecodedFrame annotatedFrame =
                      makeAnnotatedEncodeFrame(current.decodedFrame, current.result, config.visual);
                  totalOverlayMs += Ms(Clock::now() - overlayStart).count();
                  const auto encodeStart = Clock::now();
                  annotatedVideoEncoder->encodeDecodedFrame(annotatedFrame, current.pts);
                  totalAnnotatedEncodeMs += Ms(Clock::now() - encodeStart).count();
                  ++timedAnnotatedFrames;
                } else {
                  annotatedVideoEncoder->encode(drawnImage, current.pts);
                }
              }
            }
            (void)drawnImage;
            visualizer->show();
          } else if (annotatedVideoEncoder && useRgaAnnotatedOutput) {
            if (!annotatedVideoEncoderInitialized) {
              EncoderConfig encCfg;
              encCfg.outputPath = annotatedOutputPath;
              encCfg.codec = config.encoderCodec;
              encCfg.fps = outputEncoderFps;
              encCfg.fpsNum = resolveEncoderFpsNum(config, sourceVideoInfo);
              encCfg.fpsDen = resolveEncoderFpsDen(config, sourceVideoInfo);
              encCfg.width = current.decodedFrame.width;
              encCfg.height = current.decodedFrame.height;
              encCfg.horStride = current.decodedFrame.horizontalStride > 0
                  ? current.decodedFrame.horizontalStride
                  : current.decodedFrame.width;
              encCfg.verStride = current.decodedFrame.verticalStride > 0
                  ? current.decodedFrame.verticalStride
                  : current.decodedFrame.height;
              encCfg.bitrate = resolveEncoderBitrate(config, encCfg.width, encCfg.height, outputEncoderFps);
              encCfg.inputFormat = PixelFormat::kNv12;
              encCfg.lowLatency = config.encoderLowLatency;
              if (config.verbose) {
                std::cerr << "[PIPELINE] init_annotated_encoder backend=" << annotatedVideoEncoder->name()
                          << " codec=" << encCfg.codec
                          << " path=" << encCfg.outputPath
                          << " size=" << encCfg.width << "x" << encCfg.height
                          << " stride=" << encCfg.horStride << "x" << encCfg.verStride
                          << " fps=" << encCfg.fpsNum << "/" << encCfg.fpsDen
                          << " bitrate=" << encCfg.bitrate
                          << " input_format=NV12\n";
              }
              annotatedVideoEncoder->init(encCfg);
              annotatedVideoEncoderInitialized = true;
            }
            if (keepEncodedFrame) {
              const auto overlayStart = Clock::now();
              DecodedFrame annotatedFrame =
                  makeAnnotatedEncodeFrame(current.decodedFrame, current.result, config.visual);
              totalOverlayMs += Ms(Clock::now() - overlayStart).count();
              const auto encodeStart = Clock::now();
              annotatedVideoEncoder->encodeDecodedFrame(annotatedFrame, current.pts);
              totalAnnotatedEncodeMs += Ms(Clock::now() - encodeStart).count();
              ++timedAnnotatedFrames;
            }
          }

          ++nextIndex;
        }
      }

      if (encoder) {
        if (!encoderInitialized) {
          throw std::runtime_error(
              "encoder-output requested, but no decodable frame with a valid dma fd reached the output stage");
        }
        encoder->flush();
      }
      if (annotatedVideoEncoder) {
        if (!annotatedVideoEncoderInitialized) {
          throw std::runtime_error(
              "annotated output requested, but no frame reached the annotated video encoder");
        }
        annotatedVideoEncoder->flush();
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
    bool loggedFirstDecodedFrame = false;
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

        if (config.verbose && !loggedFirstDecodedFrame) {
          loggedFirstDecodedFrame = true;
          std::cerr << "[PIPELINE] first_decoded_frame"
                    << " size=" << decodedFrame->width << "x" << decodedFrame->height
                    << " stride=" << decodedFrame->horizontalStride << "x" << decodedFrame->verticalStride
                    << " chroma_stride=" << decodedFrame->chromaStride
                    << " format=" << (decodedFrame->format == PixelFormat::kNv12 ? "NV12" : "unknown")
                    << " native_format=" << decodedFrame->nativeFormat
                    << " on_device=" << (decodedFrame->isOnDevice ? "true" : "false")
                    << " dma_fd=" << decodedFrame->dmaFd
                    << "\n";
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
            PreprocessOptions{config.letterbox, 114, !config.rknnZeroCopy});
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
    try {
      std::rethrow_exception(workerError);
    } catch (const std::exception& error) {
      std::cerr << "\n[ERROR] Pipeline failed: " << error.what() << '\n';
    } catch (...) {
      std::cerr << "\n[ERROR] Pipeline failed with an unknown error\n";
    }
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(1);
  }

  // Some board-side Rockchip library combinations still crash during process
  // teardown after a fully successful run. Once all work is completed and no
  // error is pending, exit immediately to avoid destructing backend objects.
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(0);
}
