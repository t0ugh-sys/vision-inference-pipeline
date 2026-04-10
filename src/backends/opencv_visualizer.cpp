#include "visualizer.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

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

cv::Rect clampRect(const BoundingBox& box, int width, int height) {
  const int x1 = std::clamp(static_cast<int>(box.x1), 0, width - 1);
  const int y1 = std::clamp(static_cast<int>(box.y1), 0, height - 1);
  const int x2 = std::clamp(static_cast<int>(box.x2), 0, width - 1);
  const int y2 = std::clamp(static_cast<int>(box.y2), 0, height - 1);
  return cv::Rect(cv::Point(x1, y1), cv::Point(std::max(x1 + 1, x2), std::max(y1 + 1, y2)));
}

}  // namespace

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
      writer_.open(
          config_.outputVideo,
          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
          30.0,
          cv::Size(width_, height_));
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
    cv::Mat rgb = rgbImageToMat(frame);

    for (const auto& box : result.boxes) {
      const cv::Scalar color = getColor(box.classId);
      const cv::Rect rect = clampRect(box, rgb.cols, rgb.rows);
      cv::rectangle(rgb, rect, color, static_cast<int>(config_.bboxThickness));

      std::string label;
      if (config_.showLabel && !box.label.empty()) {
        label = box.label;
      }
      if (config_.showConf) {
        if (!label.empty()) {
          label += " ";
        }
        label += std::to_string(static_cast<int>(box.score * 100.0f)) + "%";
      }

      if (!label.empty()) {
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(
            label,
            cv::FONT_HERSHEY_SIMPLEX,
            config_.fontScale,
            1,
            &baseline);
        const int text_top = std::max(0, rect.y - text_size.height - 8);
        cv::rectangle(
            rgb,
            cv::Point(rect.x, text_top),
            cv::Point(std::min(rect.x + text_size.width + 8, rgb.cols - 1), rect.y),
            color,
            cv::FILLED);
        cv::putText(
            rgb,
            label,
            cv::Point(rect.x + 4, std::max(text_size.height, rect.y - 4)),
            cv::FONT_HERSHEY_SIMPLEX,
            config_.fontScale,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);
      }
    }

    display_image_ = rgb.clone();
    if (writer_.isOpened()) {
      cv::Mat bgr;
      cv::cvtColor(display_image_, bgr, cv::COLOR_RGB2BGR);
      writer_.write(bgr);
    }

    RgbImage output;
    output.width = display_image_.cols;
    output.height = display_image_.rows;
    output.data.resize(static_cast<std::size_t>(display_image_.total() * display_image_.elemSize()));
    std::memcpy(output.data.data(), display_image_.data, output.data.size());
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
    width_ = 0;
    height_ = 0;
  }

  std::string name() const override { return "OpenCV"; }

  bool isAvailable() const override { return true; }

 private:
  cv::Scalar getColor(int classId) const {
    static const cv::Scalar colors[] = {
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(128, 0, 128),
        cv::Scalar(255, 165, 0),
        cv::Scalar(128, 128, 0),
        cv::Scalar(0, 128, 128),
    };
    return colors[std::abs(classId) % 10];
  }

  int width_ = 0;
  int height_ = 0;
  VisualConfig config_;
  cv::VideoWriter writer_;
  cv::Mat display_image_;
  std::string window_name_;
};

class DummyVisualizer : public IVisualizer {
 public:
  void init(int width, int height, const VisualConfig& config) override {
    (void)width;
    (void)height;
    (void)config;
  }

  RgbImage draw(const RgbImage& frame, const DetectionResult& result) override {
    (void)result;
    return frame;
  }

  void show() override {}
  void close() override {}

  std::string name() const override { return "Dummy"; }
  bool isAvailable() const override { return false; }
};

std::unique_ptr<IVisualizer> createVisualizer() {
#ifdef OpenCV_FOUND
  return std::make_unique<OpenCVVisualizer>();
#else
  return std::make_unique<DummyVisualizer>();
#endif
}
