#include "backends/trt_infer.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

void checkTrtStatus(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void checkCudaStatus(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
struct TrtDestroy {
  void operator()(T* value) const {
    if (value != nullptr) {
      delete value;
    }
  }
};

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    (void)severity;
    (void)message;
  }
};

Logger gLogger;

std::size_t dimsElementCount(const nvinfer1::Dims& dims) {
  std::size_t count = 1;
  for (int index = 0; index < dims.nbDims; ++index) {
    if (dims.d[index] <= 0) {
      throw std::runtime_error("TensorRT binding has dynamic or invalid dimensions");
    }
    count *= static_cast<std::size_t>(dims.d[index]);
  }
  return count;
}

void packRgbToNchw(
    const RgbImage& image,
    int channels,
    std::vector<std::uint8_t>& destination) {
  const std::size_t plane_size = static_cast<std::size_t>(image.width * image.height);
  destination.resize(plane_size * static_cast<std::size_t>(channels));

  for (std::size_t index = 0; index < plane_size; ++index) {
    const std::size_t src = index * 3;
    destination[index] = image.data[src];
    if (channels > 1) {
      destination[plane_size + index] = image.data[src + 1];
    }
    if (channels > 2) {
      destination[plane_size * 2 + index] = image.data[src + 2];
    }
  }
}

}  // namespace

TrtInfer::~TrtInfer() {
  close();
}

void TrtInfer::open(const ModelConfig& config) {
  close();
  loadEngine(config.modelPath);
}

std::vector<float> TrtInfer::infer(const RgbImage& image) {
  if (!context_ || !engine_) {
    throw std::runtime_error("TensorRT backend is not initialized");
  }
  if (image.width != input_width_ || image.height != input_height_) {
    throw std::runtime_error("RGB image size does not match TensorRT input");
  }
  if (image.data.size() != input_bytes_) {
    throw std::runtime_error("RGB image buffer size does not match TensorRT input buffer");
  }

  const void* host_input = image.data.data();
  if (input_is_nchw_) {
    packRgbToNchw(image, input_channels_, host_input_buffer_);
    host_input = host_input_buffer_.data();
  }

  checkCudaStatus(
      cudaMemcpy(input_buffer_, host_input, input_bytes_, cudaMemcpyHostToDevice),
      "Failed to copy TensorRT input to device");

  std::vector<void*> bindings(engine_->getNbBindings(), nullptr);
  bindings[input_binding_] = input_buffer_;
  bindings[output_binding_] = output_buffer_;

  checkTrtStatus(context_->executeV2(bindings.data()), "TensorRT execute failed");

  std::vector<float> result(output_elements_);
  checkCudaStatus(
      cudaMemcpy(
          result.data(),
          output_buffer_,
          output_elements_ * sizeof(float),
          cudaMemcpyDeviceToHost),
      "Failed to copy TensorRT output to host");

  return result;
}

void TrtInfer::loadEngine(const std::string& path) {
  checkCudaStatus(cudaSetDevice(gpu_id_), "Failed to set CUDA device");

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open TensorRT engine: " + path);
  }

  const auto size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("TensorRT engine file is empty: " + path);
  }
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(static_cast<std::size_t>(size));
  file.read(buffer.data(), size);
  file.close();

  std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime(
      nvinfer1::createInferRuntime(gLogger));
  checkTrtStatus(runtime != nullptr, "Failed to create TensorRT runtime");

  engine_.reset(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  checkTrtStatus(engine_ != nullptr, "Failed to deserialize TensorRT engine");

  context_.reset(engine_->createExecutionContext());
  checkTrtStatus(context_ != nullptr, "Failed to create TensorRT execution context");
  checkTrtStatus(engine_->getNbBindings() >= 2, "TensorRT engine must have at least one input and one output");

  input_binding_ = 0;
  output_binding_ = 1;
  for (int index = 0; index < engine_->getNbBindings(); ++index) {
    if (engine_->bindingIsInput(index)) {
      input_binding_ = static_cast<std::size_t>(index);
    } else {
      output_binding_ = static_cast<std::size_t>(index);
      break;
    }
  }

  const nvinfer1::Dims input_dims = engine_->getBindingDimensions(static_cast<int>(input_binding_));
  checkTrtStatus(input_dims.nbDims == 4, "TensorRT input must be a 4D tensor");

  if (input_dims.d[1] > 0 && input_dims.d[1] <= 4) {
    input_is_nchw_ = true;
    input_channels_ = input_dims.d[1];
    input_height_ = input_dims.d[2];
    input_width_ = input_dims.d[3];
  } else if (input_dims.d[3] > 0 && input_dims.d[3] <= 4) {
    input_is_nchw_ = false;
    input_height_ = input_dims.d[1];
    input_width_ = input_dims.d[2];
    input_channels_ = input_dims.d[3];
  } else {
    throw std::runtime_error("Unsupported TensorRT input layout");
  }

  checkTrtStatus(
      input_channels_ == 3,
      "TensorRT input channel count must be 3 for the current RGB pipeline");
  input_bytes_ = dimsElementCount(input_dims) * sizeof(std::uint8_t);

  const nvinfer1::Dims output_dims = context_->getBindingDimensions(static_cast<int>(output_binding_));
  output_elements_ = dimsElementCount(output_dims);

  checkCudaStatus(cudaMalloc(&input_buffer_, input_bytes_), "Failed to allocate TensorRT input buffer");
  checkCudaStatus(
      cudaMalloc(&output_buffer_, output_elements_ * sizeof(float)),
      "Failed to allocate TensorRT output buffer");
}

std::size_t TrtInfer::getOutputElementCount() const {
  return output_elements_;
}

void TrtInfer::releaseBuffers() {
  if (input_buffer_ != nullptr) {
    cudaFree(input_buffer_);
    input_buffer_ = nullptr;
  }
  if (output_buffer_ != nullptr) {
    cudaFree(output_buffer_);
    output_buffer_ = nullptr;
  }
}

void TrtInfer::close() {
  releaseBuffers();
  context_.reset();
  engine_.reset();
  input_width_ = 0;
  input_height_ = 0;
  input_channels_ = 0;
  input_is_nchw_ = true;
  input_binding_ = 0;
  output_binding_ = 1;
  input_bytes_ = 0;
  output_elements_ = 0;
  host_input_buffer_.clear();
}
