#include "backends/rknn_infer.hpp"

#include "rknn_api.h"

#include <cstring>
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

TensorDataType toTensorDataType(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32: return TensorDataType::kFloat32;
    case RKNN_TENSOR_UINT8: return TensorDataType::kUint8;
    case RKNN_TENSOR_INT8: return TensorDataType::kInt8;
    case RKNN_TENSOR_INT32: return TensorDataType::kInt32;
    default: return TensorDataType::kUnknown;
  }
}

TensorQuantizationType toQuantizationType(rknn_tensor_qnt_type type) {
  switch (type) {
    case RKNN_TENSOR_QNT_DFP: return TensorQuantizationType::kDfp;
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: return TensorQuantizationType::kAffineAsymmetric;
    case RKNN_TENSOR_QNT_NONE:
    default: return TensorQuantizationType::kNone;
  }
}

std::size_t tensorElementCount(const rknn_tensor_attr& attr) {
  std::size_t count = 1;
  for (std::uint32_t i = 0; i < attr.n_dims; ++i) {
    count *= static_cast<std::size_t>(attr.dims[i]);
  }
  return count;
}

std::size_t tensorTypeBytes(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32: return sizeof(float);
    case RKNN_TENSOR_INT32: return sizeof(std::int32_t);
    case RKNN_TENSOR_UINT8: return sizeof(std::uint8_t);
    case RKNN_TENSOR_INT8: return sizeof(std::int8_t);
    default: return sizeof(std::uint8_t);
  }
}

std::string toTensorLayout(rknn_tensor_format fmt) {
  switch (fmt) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    default: return "UNSPECIFIED";
  }
}

std::vector<std::int64_t> toShape(const rknn_tensor_attr& attr) {
  std::vector<std::int64_t> shape;
  shape.reserve(attr.n_dims);
  for (std::uint32_t i = 0; i < attr.n_dims; ++i) {
    shape.push_back(attr.dims[i]);
  }
  return shape;
}

}  // namespace

namespace {

class RknnOutputGuard {
 public:
  RknnOutputGuard(rknn_context context, std::vector<rknn_output>& outputs)
      : context_(context), outputs_(outputs), armed_(false) {}

  void arm() { armed_ = true; }

  ~RknnOutputGuard() {
    if (armed_) {
      rknn_outputs_release(context_, static_cast<uint32_t>(outputs_.size()), outputs_.data());
    }
  }

 private:
  rknn_context context_;
  std::vector<rknn_output>& outputs_;
  bool armed_;
};

class RknnTensorMemGuard {
 public:
  explicit RknnTensorMemGuard(rknn_context context) : context_(context) {}

  void reset(rknn_tensor_mem* mem) { mem_ = mem; }

  ~RknnTensorMemGuard() {
    if (mem_ != nullptr) {
      rknn_destroy_mem(context_, mem_);
    }
  }

 private:
  rknn_context context_;
  rknn_tensor_mem* mem_ = nullptr;
};

}  // namespace

RknnInfer::~RknnInfer() {
  close();
}

void RknnInfer::open(const ModelConfig& config) {
  close();
  model_data_ = readModelFile(config.modelPath);
  checkRknnStatus(rknn_init(&context_, model_data_.data(), model_data_.size(), 0, nullptr), "rknn_init failed");
  queryTensorInfo();
}

InferenceOutput RknnInfer::infer(const RgbImage& image) {
  if (context_ == 0) {
    throw std::runtime_error("RKNN backend is not initialized");
  }
  if (image.width != input_width_ || image.height != input_height_) {
    throw std::runtime_error("RGB image size does not match RKNN input tensor");
  }

  const bool canUseFdInput =
      is_nhwc_ &&
      has_native_input_attr_ &&
      image.format == PixelFormat::kRgb888 &&
      image.dmaFd >= 0 &&
      image.wstride == static_cast<int>(native_input_attr_.w_stride) &&
      image.dmaSize >= static_cast<std::size_t>(
          native_input_attr_.size_with_stride != 0 ? native_input_attr_.size_with_stride : native_input_attr_.size);

  std::vector<std::uint8_t> inputBuffer;
  RknnTensorMemGuard tensorMemGuard(context_);

  if (canUseFdInput) {
    const uint32_t nativeInputBytes =
        native_input_attr_.size_with_stride != 0 ? native_input_attr_.size_with_stride : native_input_attr_.size;
    auto* mem = rknn_create_mem_from_fd(context_, image.dmaFd, nullptr, nativeInputBytes, 0);
    if (mem == nullptr) {
      throw std::runtime_error("rknn_create_mem_from_fd failed");
    }
    tensorMemGuard.reset(mem);
    checkRknnStatus(rknn_set_io_mem(context_, mem, &native_input_attr_), "rknn_set_io_mem failed");
  } else {
    if (is_nhwc_) {
      inputBuffer = image.data;
    } else {
      const std::size_t planeSize = static_cast<std::size_t>(image.width * image.height);
      inputBuffer.resize(planeSize * 3);
      for (std::size_t i = 0; i < planeSize; ++i) {
        const std::size_t src = i * 3;
        inputBuffer[i] = image.data[src];
        inputBuffer[planeSize + i] = image.data[src + 1];
        inputBuffer[planeSize * 2 + i] = image.data[src + 2];
      }
    }

    rknn_input input = {};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.size = static_cast<uint32_t>(inputBuffer.size());
    input.fmt = is_nhwc_ ? RKNN_TENSOR_NHWC : RKNN_TENSOR_NCHW;
    input.buf = inputBuffer.data();

    checkRknnStatus(rknn_inputs_set(context_, 1, &input), "rknn_inputs_set failed");
  }

  checkRknnStatus(rknn_run(context_, nullptr), "rknn_run failed");

  std::vector<rknn_output> outputs(output_templates_.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].want_float = output_templates_[i].dataType == TensorDataType::kFloat32 ? 1 : 0;
  }
  RknnOutputGuard outputGuard(context_, outputs);

  checkRknnStatus(
      rknn_outputs_get(context_, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr),
      "rknn_outputs_get failed");
  outputGuard.arm();

  InferenceOutput result = output_templates_;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    result[i].data.clear();
    result[i].rawData.resize(outputs[i].size);
    std::memcpy(result[i].rawData.data(), outputs[i].buf, outputs[i].size);
    if (outputs[i].want_float != 0) {
      const auto count = outputs[i].size / sizeof(float);
      const auto* data = static_cast<const float*>(outputs[i].buf);
      result[i].data.assign(data, data + count);
    }
  }
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
  rknn_input_output_num ioNum = {};
  checkRknnStatus(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)), "RKNN_QUERY_IN_OUT_NUM failed");

  input_attr_ = makeTensorAttr(0);
  checkRknnStatus(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)), "RKNN_QUERY_INPUT_ATTR failed");

  native_input_attr_ = makeTensorAttr(0);
  has_native_input_attr_ =
      (rknn_query(context_, RKNN_QUERY_NATIVE_INPUT_ATTR, &native_input_attr_, sizeof(native_input_attr_)) == RKNN_SUCC);

  if (input_attr_.fmt == RKNN_TENSOR_NHWC) {
    is_nhwc_ = true;
    input_height_ = input_attr_.dims[1];
    input_width_ = input_attr_.dims[2];
    input_channels_ = input_attr_.dims[3];
  } else if (input_attr_.fmt == RKNN_TENSOR_NCHW) {
    is_nhwc_ = false;
    input_channels_ = input_attr_.dims[1];
    input_height_ = input_attr_.dims[2];
    input_width_ = input_attr_.dims[3];
  } else {
    throw std::runtime_error("Unsupported RKNN input tensor format");
  }

  output_templates_.clear();
  output_templates_.reserve(ioNum.n_output);
  for (std::uint32_t i = 0; i < ioNum.n_output; ++i) {
    rknn_tensor_attr outputAttr = makeTensorAttr(i);
    checkRknnStatus(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &outputAttr, sizeof(outputAttr)), "RKNN_QUERY_OUTPUT_ATTR failed");

    InferenceTensor tensor;
    tensor.name = outputAttr.name[0] != '\0' ? outputAttr.name : ("output_" + std::to_string(i));
    tensor.layout = toTensorLayout(outputAttr.fmt);
    tensor.shape = toShape(outputAttr);
    tensor.dataType = toTensorDataType(outputAttr.type);
    tensor.quantization = toQuantizationType(outputAttr.qnt_type);
    tensor.zeroPoint = outputAttr.zp;
    tensor.fractionalLength = outputAttr.fl;
    tensor.scale = outputAttr.scale;
    tensor.rawData.reserve(tensorElementCount(outputAttr) * tensorTypeBytes(outputAttr.type));
    output_templates_.push_back(std::move(tensor));
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
  has_native_input_attr_ = false;
  input_attr_ = {};
  native_input_attr_ = {};
  output_templates_.clear();
}
