#include "visualizer.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int COLOR_BLUE = 0xFF0000FFU;
constexpr unsigned int COLOR_RED = 0xFFFF0000U;
constexpr unsigned int COLOR_WHITE = 0xFFFFFFFFU;
constexpr int kModelZooBoxThickness = 3;
constexpr int kModelZooFontPixelSize = 10;

struct RgbColor {
  std::uint8_t r = 0;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
};

constexpr RgbColor kUltralyticsPalette[] = {
    {4, 42, 255},   {11, 219, 235}, {243, 243, 243}, {0, 223, 183},  {17, 31, 104},
    {255, 111, 221}, {255, 68, 79}, {204, 237, 0},   {0, 243, 68},   {189, 0, 255},
    {0, 180, 255},  {221, 0, 186}, {0, 255, 255},   {38, 192, 0},   {1, 255, 179},
    {125, 36, 255}, {123, 0, 104}, {255, 27, 108},  {252, 109, 47}, {162, 255, 11},
};

int clampValue(float value, int minValue, int maxValue) {
  if (value < static_cast<float>(minValue)) {
    return minValue;
  }
  if (value > static_cast<float>(maxValue)) {
    return maxValue;
  }
  return static_cast<int>(value);
}

unsigned int convertColorRgb888(unsigned int srcColor) {
  unsigned int dstColor = 0;
  unsigned char* src = reinterpret_cast<unsigned char*>(&srcColor);
  unsigned char* dst = reinterpret_cast<unsigned char*>(&dstColor);
  const unsigned char r = src[2];
  const unsigned char g = src[1];
  const unsigned char b = src[0];
  dst[0] = r;
  dst[1] = g;
  dst[2] = b;
  return dstColor;
}

unsigned int encodeRgb888(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  unsigned int color = 0;
  auto* dst = reinterpret_cast<unsigned char*>(&color);
  dst[0] = r;
  dst[1] = g;
  dst[2] = b;
  return color;
}

RgbColor ultralyticsColorForClass(int classId) {
  const std::size_t paletteSize = sizeof(kUltralyticsPalette) / sizeof(kUltralyticsPalette[0]);
  const std::size_t index =
      static_cast<std::size_t>(classId >= 0 ? classId : 0) % paletteSize;
  return kUltralyticsPalette[index];
}

unsigned int ultralyticsTextColor(const RgbColor& background) {
  const int luminance = static_cast<int>(background.r) + static_cast<int>(background.g) + static_cast<int>(background.b);
  return luminance >= 600 ? encodeRgb888(16, 16, 16) : COLOR_WHITE;
}

int ultralyticsLineWidth(const RgbImage& image) {
  return std::max(static_cast<int>(std::lround((image.width + image.height) * 0.003 / 2.0)), 2);
}

void drawRectangleC3(
    unsigned char* pixels,
    int width,
    int height,
    int rx,
    int ry,
    int rw,
    int rh,
    unsigned int color,
    int thickness) {
  const unsigned char* penColor = reinterpret_cast<unsigned char*>(&color);
  const int stride = width * 3;

  if (thickness == -1) {
    for (int y = ry; y < ry + rh; ++y) {
      if (y < 0) {
        continue;
      }
      if (y >= height) {
        break;
      }
      unsigned char* p = pixels + stride * y;
      for (int x = rx; x < rx + rw; ++x) {
        if (x < 0) {
          continue;
        }
        if (x >= width) {
          break;
        }
        p[x * 3 + 0] = penColor[0];
        p[x * 3 + 1] = penColor[1];
        p[x * 3 + 2] = penColor[2];
      }
    }
    return;
  }

  const int t0 = thickness / 2;
  const int t1 = thickness - t0;

  for (int y = ry - t0; y < ry + t1; ++y) {
    if (y < 0) {
      continue;
    }
    if (y >= height) {
      break;
    }
    unsigned char* p = pixels + stride * y;
    for (int x = rx - t0; x < rx + rw + t1; ++x) {
      if (x < 0) {
        continue;
      }
      if (x >= width) {
        break;
      }
      p[x * 3 + 0] = penColor[0];
      p[x * 3 + 1] = penColor[1];
      p[x * 3 + 2] = penColor[2];
    }
  }

  for (int y = ry + rh - t0; y < ry + rh + t1; ++y) {
    if (y < 0) {
      continue;
    }
    if (y >= height) {
      break;
    }
    unsigned char* p = pixels + stride * y;
    for (int x = rx - t0; x < rx + rw + t1; ++x) {
      if (x < 0) {
        continue;
      }
      if (x >= width) {
        break;
      }
      p[x * 3 + 0] = penColor[0];
      p[x * 3 + 1] = penColor[1];
      p[x * 3 + 2] = penColor[2];
    }
  }

  for (int x = rx - t0; x < rx + t1; ++x) {
    if (x < 0) {
      continue;
    }
    if (x >= width) {
      break;
    }
    for (int y = ry + t1; y < ry + rh - t0; ++y) {
      if (y < 0) {
        continue;
      }
      if (y >= height) {
        break;
      }
      unsigned char* p = pixels + stride * y;
      p[x * 3 + 0] = penColor[0];
      p[x * 3 + 1] = penColor[1];
      p[x * 3 + 2] = penColor[2];
    }
  }

  for (int x = rx + rw - t0; x < rx + rw + t1; ++x) {
    if (x < 0) {
      continue;
    }
    if (x >= width) {
      break;
    }
    for (int y = ry + t1; y < ry + rh - t0; ++y) {
      if (y < 0) {
        continue;
      }
      if (y >= height) {
        break;
      }
      unsigned char* p = pixels + stride * y;
      p[x * 3 + 0] = penColor[0];
      p[x * 3 + 1] = penColor[1];
      p[x * 3 + 2] = penColor[2];
    }
  }
}

void drawTextC3(
    unsigned char* pixels,
    int width,
    int height,
    const char* text,
    int x,
    int y,
    int fontPixelSize,
    unsigned int color) {
  cv::Mat rgb(height, width, CV_8UC3, pixels);
  const cv::Scalar drawColor(
      static_cast<double>(color & 0xFFu),
      static_cast<double>((color >> 8) & 0xFFu),
      static_cast<double>((color >> 16) & 0xFFu));
  const double fontScale = std::max(0.3, static_cast<double>(fontPixelSize) / 20.0);
  const int thickness = std::max(1, fontPixelSize / 6);
  cv::putText(
      rgb,
      text,
      cv::Point(x, std::max(fontPixelSize, y + fontPixelSize)),
      cv::FONT_HERSHEY_SIMPLEX,
      fontScale,
      drawColor,
      thickness,
      cv::LINE_AA);
}

void drawRectangle(
    RgbImage& image,
    int x,
    int y,
    int width,
    int height,
    unsigned int color,
    int thickness) {
  if (image.data.empty() || image.width <= 0 || image.height <= 0) {
    return;
  }
  const unsigned int drawColor = convertColorRgb888(color);
  drawRectangleC3(image.data.data(), image.width, image.height, x, y, width, height, drawColor, thickness);
}

void drawText(
    RgbImage& image,
    const char* text,
    int x,
    int y,
    unsigned int color,
    int fontPixelSize) {
  if (image.data.empty() || image.width <= 0 || image.height <= 0) {
    return;
  }
  const unsigned int drawColor = convertColorRgb888(color);
  drawTextC3(image.data.data(), image.width, image.height, text, x, y, fontPixelSize, drawColor);
}

void drawYoloLabelBox(
    RgbImage& image,
    const char* text,
    int anchorX,
    int anchorY,
    unsigned int backgroundColor,
    unsigned int textColor,
    int fontPixelSize) {
  if (image.data.empty() || image.width <= 0 || image.height <= 0 || text == nullptr || text[0] == '\0') {
    return;
  }

  cv::Mat rgb(image.height, image.width, CV_8UC3, image.data.data());
  const double fontScale = std::max(0.3, static_cast<double>(fontPixelSize) / 20.0);
  const int textThickness = std::max(1, fontPixelSize / 6);
  int baseline = 0;
  const cv::Size textSize =
      cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, textThickness, &baseline);

  const int padX = std::max(4, fontPixelSize / 2);
  const int padY = std::max(3, fontPixelSize / 3);
  int labelX = std::clamp(anchorX, 0, std::max(0, image.width - 1));
  int labelY = anchorY - textSize.height - baseline - padY * 2;
  if (labelY < 0) {
    labelY = std::clamp(anchorY + 1, 0, std::max(0, image.height - 1));
  }

  const int labelWidth = std::min(image.width - labelX, textSize.width + padX * 2);
  const int labelHeight = std::min(std::max(1, image.height - labelY), textSize.height + baseline + padY * 2);
  if (labelWidth <= 0 || labelHeight <= 0) {
    return;
  }

  const cv::Rect labelRect(labelX, labelY, labelWidth, labelHeight);
  const cv::Scalar bg(
      static_cast<double>(backgroundColor & 0xFFu),
      static_cast<double>((backgroundColor >> 8) & 0xFFu),
      static_cast<double>((backgroundColor >> 16) & 0xFFu));
  const cv::Scalar fg(
      static_cast<double>(textColor & 0xFFu),
      static_cast<double>((textColor >> 8) & 0xFFu),
      static_cast<double>((textColor >> 16) & 0xFFu));

  cv::rectangle(rgb, labelRect, bg, cv::FILLED);
  cv::putText(
      rgb,
      text,
      cv::Point(labelX + padX, labelY + padY + textSize.height),
      cv::FONT_HERSHEY_SIMPLEX,
      fontScale,
      fg,
      textThickness,
      cv::LINE_AA);
}

void drawYoloBoxLabel(
    RgbImage& image,
    int x1,
    int y1,
    int x2,
    int y2,
    const char* text,
    const RgbColor& classColor) {
  if (image.data.empty() || image.width <= 0 || image.height <= 0) {
    return;
  }

  cv::Mat rgb(image.height, image.width, CV_8UC3, image.data.data());
  const int lineWidth = ultralyticsLineWidth(image);
  const int textThickness = std::max(lineWidth - 1, 1);
  const double fontScale = static_cast<double>(lineWidth) / 3.0;
  const cv::Scalar color(
      static_cast<double>(classColor.r),
      static_cast<double>(classColor.g),
      static_cast<double>(classColor.b));
  cv::rectangle(rgb, cv::Point(x1, y1), cv::Point(x2, y2), color, lineWidth, cv::LINE_AA);

  if (text == nullptr || text[0] == '\0') {
    return;
  }

  const unsigned int textColor = ultralyticsTextColor(classColor);
  const cv::Scalar txtColor(
      static_cast<double>(textColor & 0xFFu),
      static_cast<double>((textColor >> 8) & 0xFFu),
      static_cast<double>((textColor >> 16) & 0xFFu));
  int baseline = 0;
  cv::Size textSize = cv::getTextSize(text, 0, fontScale, textThickness, &baseline);
  textSize.height += 3;
  bool outside = y1 >= textSize.height;
  int labelX = x1;
  if (labelX > image.width - textSize.width) {
    labelX = std::max(0, image.width - textSize.width);
  }
  cv::Point p1(labelX, y1);
  cv::Point p2(labelX + textSize.width, outside ? y1 - textSize.height : y1 + textSize.height);
  cv::rectangle(rgb, p1, p2, color, cv::FILLED, cv::LINE_AA);
  cv::putText(
      rgb,
      text,
      cv::Point(labelX, outside ? y1 - 2 : y1 + textSize.height - 1),
      0,
      fontScale,
      txtColor,
      textThickness,
      cv::LINE_AA);
}

cv::Mat rgbImageToMat(const RgbImage& frame) {
  if (frame.width <= 0 || frame.height <= 0) {
    throw std::runtime_error("Visualizer received an invalid frame size");
  }
  if (frame.data.size() != static_cast<std::size_t>(frame.width * frame.height * 3)) {
    throw std::runtime_error("Visualizer received an invalid RGB frame buffer");
  }

  cv::Mat rgb(frame.height, frame.width, CV_8UC3, const_cast<std::uint8_t*>(frame.data.data()));
  return rgb.clone();
}

class OpenCVVisualizer : public IVisualizer {
 public:
  OpenCVVisualizer() = default;
  ~OpenCVVisualizer() override { close(); }

  void init(int width, int height, const VisualConfig& config) override {
    close();
    width_ = width;
    height_ = height;
    config_ = config;

    if (!config_.outputVideo.empty()) {
      writer_.open(config_.outputVideo, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, cv::Size(width_, height_));
      if (!writer_.isOpened()) {
        throw std::runtime_error("Failed to open video writer: " + config_.outputVideo);
      }
    }

    if (config_.display) {
      window_name_ = "video_pipeline";
      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    }
  }

  RgbImage draw(const RgbImage& frame, const DetectionResult& result) override {
    RgbImage output = frame;
    for (const auto& box : result.boxes) {
      const int x1 = clampValue(box.x1, 0, std::max(0, output.width - 1));
      const int y1 = clampValue(box.y1, 0, std::max(0, output.height - 1));
      const int x2 = clampValue(box.x2, 0, std::max(0, output.width - 1));
      const int y2 = clampValue(box.y2, 0, std::max(0, output.height - 1));
      const int w = std::max(1, x2 - x1);
      const int h = std::max(1, y2 - y1);

      const RgbColor classColor = ultralyticsColorForClass(box.classId);
      const unsigned int classColorValue = encodeRgb888(classColor.r, classColor.g, classColor.b);

      char text[256] = {};
      if (config_.showLabel && !box.label.empty() && config_.showConf) {
        std::snprintf(text, sizeof(text), "%s %.2f", box.label.c_str(), box.score);
      } else if (config_.showLabel && !box.label.empty()) {
        std::snprintf(text, sizeof(text), "%s", box.label.c_str());
      } else if (config_.showConf) {
        std::snprintf(text, sizeof(text), "%.2f", box.score);
      }

      if (config_.style == VisualStyle::kYolo) {
        drawYoloBoxLabel(output, x1, y1, x2, y2, text, classColor);
      } else {
        drawRectangle(output, x1, y1, w, h, COLOR_BLUE, kModelZooBoxThickness);
        if (text[0] != '\0') {
          drawText(output, text, x1, y1 - 20, COLOR_RED, kModelZooFontPixelSize);
        }
      }
    }

    display_image_ = rgbImageToMat(output);
    if (writer_.isOpened()) {
      cv::Mat bgr;
      cv::cvtColor(display_image_, bgr, cv::COLOR_RGB2BGR);
      writer_.write(bgr);
    }
    return output;
  }

  void show() override {
    if (config_.display && !display_image_.empty()) {
      cv::Mat bgr;
      cv::cvtColor(display_image_, bgr, cv::COLOR_RGB2BGR);
      cv::imshow(window_name_, bgr);
      cv::waitKey(1);
    }
  }

  void close() override {
    if (writer_.isOpened()) {
      writer_.release();
    }
    if (!window_name_.empty()) {
      cv::destroyWindow(window_name_);
      window_name_.clear();
    }
    display_image_.release();
  }

  std::string name() const override { return "OpenCV"; }
  bool isAvailable() const override { return true; }

 private:
  int width_ = 0;
  int height_ = 0;
  VisualConfig config_;
  cv::VideoWriter writer_;
  cv::Mat display_image_;
  std::string window_name_;
};

}  // namespace

std::unique_ptr<IVisualizer> createVisualizer() {
  return std::make_unique<OpenCVVisualizer>();
}
