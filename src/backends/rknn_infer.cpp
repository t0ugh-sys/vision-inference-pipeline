#include "backends/rknn_infer.hpp"

#include "third_party/rknn/include/rknn_api.h"

#include <fstream>
#include <stdexcept>

namespace {

void checkRknnStatus(int status, const char* message) {
  if (status != RKNN_SUCC) {
    throw std::runtime_error(message);
  }
}

rknn_tensor_attr makeTensorAttr(std::uint32_t index) {
  rknn_tensor_attr attr = {};
  attr.index = index;
  return attr;
}

}  // namespace

RknnInfer::~RknnInfer() {
  close();
}

void RknnInfer::open(const ModelConfig& config) {
  close();
  model_data_ = readModelFile(config.modelPath);

  checkRknnStatus(
      rknn_init(&context_, model_data_.data(), model_data_.size(), 0, nullptr),
      "rknn_init failed");

  queryTensorInfo();
}

std::vector<float> RknnInfer::infer(const RgbImage& image) {
  if (context_ == 0) {
    throw std::runtime_error("RKNN backend is not initialized");
  }
  if (image.width != input_width_ || image.height != input_height_) {
    throw std::runtime_error("RGB image size does not match RKNN input tensor");
  }
  if (image.data.size() != static_cast<std::size_t>(input_width_ * input_height_ * input_channels_)) {
    throw std::runtime_error("RGB image buffer size does not match RKNN input tensor");
  }

  // 准备输入数据
  std::vector<std::uint8_t> input_buffer;
  if (is_nhwc_) {
    input_buffer = image.data;  // HWC 直传
  } else {
    // NCHW 格式转换
    const std::size_t planeSize = static_cast<std::size_t>(image.width * image.height);
    input_buffer.resize(planeSize * 3);
    for (std::size_t i = 0; i < planeSize; ++i) {
      const std::size_t src = i * 3;
      input_buffer[i] = image.data[src];                         // R
      input_buffer[planeSize + i] = image.data[src + 1];         // G
      input_buffer[planeSize * 2 + i] = image.data[src + 2];     // B
    }
  }

  rknn_input input = {};
  input.index = 0;
  input.type = RKNN_TENSOR_UINT8;
  input.size = input_buffer.size();
  input.fmt = is_nhwc_ ? RKNN_TENSOR_NHWC : RKNN_TENSOR_NCHW;
  input.buf = input_buffer.data();

  checkRknnStatus(rknn_inputs_set(context_, 1, &input), "rknn_inputs_set failed");
  checkRknnStatus(rknn_run(context_, nullptr), "rknn_run failed");

  // 获取输出
  rknn_output_num output_num = {};
  checkRknnStatus(
      rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &output_num, sizeof(output_num)),
      "RKNN_QUERY_IN_OUT_NUM failed");

  std::vector<rknn_output> outputs(output_num.n_output);
  for (auto& output : outputs) {
    output.want_float = 1;
  }

  checkRknnStatus(
      rknn_outputs_get(context_, outputs.size(), outputs.data(), nullptr),
      "rknn_outputs_get failed");

  std::vector<float> result;
  for (std::uint32_t i = 0; i < outputs.size(); ++i) {
    const auto bytes = static_cast<std::uint32_t>(outputs[i].size);
    const auto count = bytes / sizeof(float);
    const auto* data = static_cast<const float*>(outputs[i].buf);
    result.insert(result.end(), data, data + count);
  }

  rknn_outputs_release(context_, outputs.size(), outputs.data());
  return result;
}

std::vector<std::uint8_t> RknnInfer::readModelFile(const std::string& path) const {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open RKNN model file: " + path);
  }

  const auto size = input.tellg();
  if (size <= 0) {
    throw std::runtime_error("RKNN model file is empty: " + path);
  }

  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

void RknnInfer::queryTensorInfo() {
  rknn_input_output_num io_num = {};
  checkRknnStatus(
      rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
      "RKNN_QUERY_IN_OUT_NUM failed");

  rknn_tensor_attr input_attr = makeTensorAttr(0);
  checkRknnStatus(
      rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr)),
      "RKNN_QUERY_INPUT_ATTR failed");

  if (input_attr.fmt == RKNN_TENSOR_NHWC) {
    is_nhwc_ = true;
    input_height_ = input_attr.dims[1];
    input_width_ = input_attr.dims[2];
    input_channels_ = input_attr.dims[3];
  } else if (input_attr.fmt == RKNN_TENSOR_NCHW) {
    is_nhwc_ = false;
    input_channels_ = input_attr.dims[1];
    input_height_ = input_attr.dims[2];
    input_width_ = input_attr.dims[3];
  } else {
    throw std::runtime_error("Unsupported RKNN input tensor format");
  }
}

void RknnInfer::close() {
  if (context_ != 0) {
    rknn_destroy(context_);
  }
  context_ = 0;
  model_data_.clear();
  input_width_ = 0;
  input_height_ = 0;
  input_channels_ = 0;
  is_nhwc_ = true;
}
