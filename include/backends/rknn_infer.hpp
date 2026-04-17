#pragma once

#include "infer_interface.hpp"
#include "pipeline_types.hpp"
#include "rknn_api.h"

#include <cstdint>
#include <vector>

class RknnInfer : public IInferenceBackend {
 public:
  RknnInfer() = default;
  ~RknnInfer() override;

  RknnInfer(const RknnInfer&) = delete;
  RknnInfer& operator=(const RknnInfer&) = delete;

  void open(const ModelConfig& config) override;
  InferenceOutput infer(const RgbImage& image) override;
  int inputWidth() const override { return input_width_; }
  int inputHeight() const override { return input_height_; }
  std::string name() const override { return "Rockchip RKNN"; }

 private:
  std::vector<std::uint8_t> readModelFile(const std::string& path) const;
  void queryTensorInfo();
  void close();

  rknn_context context_ = 0;
  std::vector<std::uint8_t> model_data_;
  int input_width_ = 0;
  int input_height_ = 0;
  int input_channels_ = 0;
  bool is_nhwc_ = true;
  bool has_native_input_attr_ = false;
  rknn_tensor_attr input_attr_ = {};
  rknn_tensor_attr native_input_attr_ = {};
  InferenceOutput output_templates_;
};
