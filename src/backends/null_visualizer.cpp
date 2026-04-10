#include "visualizer.hpp"

#include <stdexcept>

/**
 * Null Visualizer (stub for Windows without OpenCV)
 */
class NullVisualizer : public IVisualizer {
 public:
  void init(int width, int height, const VisualConfig& config) override {
    (void)width;
    (void)height;
    (void)config;
  }

  RgbImage draw(const RgbImage& frame, const DetectionResult& result) override {
    (void)frame;
    (void)result;
    // Just return the frame as-is without drawing
    return frame;
  }

  void show() override {
    // No-op
  }

  void close() override {
    // No-op
  }

  std::string name() const override { return "Null (disabled)"; }

  bool isAvailable() const override { return false; }
};

std::unique_ptr<IVisualizer> createVisualizer() {
  return std::make_unique<NullVisualizer>();
}
