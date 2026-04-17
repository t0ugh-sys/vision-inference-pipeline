#include "backends/rga_preprocessor.hpp"

extern "C" {
#include <mpp_buffer.h>
}
#include <im2d.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

// RGA hardware requires aligned stride for correct operation. 16 is a safe
// universal choice covering RGA2/RGA3 and both NV12 (min 2) and RGB_888 (min 4).
constexpr int kRgaStrideAlign = 16;
constexpr size_t kRgaBufferGroupMaxBytes = 32 * 1024 * 1024;
constexpr RK_S32 kRgaBufferGroupLimit = 4;

inline int alignUp(int value, int alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// imresize / imcvtcolor return IM_STATUS_SUCCESS (=1) on success.
void checkRgaOp(IM_STATUS status, const char* stage) {
  if (status != IM_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string("RGA ") + stage + " failed: " + imStrError_t(status));
  }
}

// imcheck returns IM_STATUS_NOERROR (=2) when parameters are valid.
void checkRgaVerify(int ret, const char* stage) {
  if (ret != IM_STATUS_NOERROR) {
    throw std::runtime_error(
        std::string("RGA imcheck before ") + stage + " failed: " +
        imStrError_t(static_cast<IM_STATUS>(ret)));
  }
}

LetterboxInfo buildLetterboxInfo(
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    bool enabled) {
  LetterboxInfo info;
  if (!enabled) {
    return info;
  }

  const float scale = std::min(
      static_cast<float>(dstWidth) / static_cast<float>(srcWidth),
      static_cast<float>(dstHeight) / static_cast<float>(srcHeight));
  info.enabled = true;
  info.scale = scale;
  info.resizedWidth = std::max(1, static_cast<int>(srcWidth * scale));
  info.resizedHeight = std::max(1, static_cast<int>(srcHeight * scale));
  info.padLeft = (dstWidth - info.resizedWidth) / 2;
  info.padTop = (dstHeight - info.resizedHeight) / 2;
  info.padRight = dstWidth - info.resizedWidth - info.padLeft;
  info.padBottom = dstHeight - info.resizedHeight - info.padTop;
  return info;
}

// Row-copy from an aligned-stride source into a packed destination.
void copyStridedToPacked(
    const std::uint8_t* src,
    int srcRowBytes,
    std::uint8_t* dst,
    int dstRowBytes,
    int height) {
  for (int y = 0; y < height; ++y) {
    std::memcpy(dst + y * dstRowBytes, src + y * srcRowBytes, dstRowBytes);
  }
}

int nv12BytesForStride(int wstride, int hstride) {
  return wstride * hstride * 3 / 2;
}

int rgb888BytesForStride(int wstride, int hstride) {
  return wstride * hstride * 3;
}

void releaseHandle(rga_buffer_handle_t& handle) {
  if (handle != 0) {
    releasebuffer_handle(handle);
    handle = 0;
  }
}

std::shared_ptr<void> makeMppBufferHandle(MppBuffer buffer) {
  return std::shared_ptr<void>(buffer, [](void* opaque) {
    if (opaque != nullptr) {
      MppBuffer buffer = opaque;
      mpp_buffer_put(buffer);
    }
  });
}

MppBuffer allocateBuffer(MppBufferGroup group, std::size_t size, const char* stage) {
  MppBuffer buffer = nullptr;
  if (mpp_buffer_get(group, &buffer, size) != MPP_OK || buffer == nullptr) {
    throw std::runtime_error(std::string("Failed to allocate DRM buffer for ") + stage);
  }
  return buffer;
}

void fillPackedRgbData(
    const std::shared_ptr<void>& nativeHandle,
    int srcWstride,
    int width,
    int height,
    std::vector<std::uint8_t>& output) {
  output.resize(static_cast<std::size_t>(width * height * 3));
  const auto* src = static_cast<const std::uint8_t*>(mpp_buffer_get_ptr(static_cast<MppBuffer>(nativeHandle.get())));
  if (src == nullptr) {
    throw std::runtime_error("Failed to map RGA output DRM buffer");
  }
  copyStridedToPacked(src, srcWstride * 3, output.data(), width * 3, height);
}

}  // namespace

RgaPreprocessor::~RgaPreprocessor() {
  if (bufferGroup_ != nullptr) {
    mpp_buffer_group_put(static_cast<MppBufferGroup>(bufferGroup_));
    bufferGroup_ = nullptr;
  }
}

void RgaPreprocessor::ensureBufferGroup() {
  if (bufferGroup_ != nullptr) {
    return;
  }
  MppBufferGroup group = nullptr;
  if (mpp_buffer_group_get_internal(
          &group,
          static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)) != MPP_OK) {
    throw std::runtime_error("mpp_buffer_group_get_internal for RGA output failed");
  }
  if (mpp_buffer_group_limit_config(group, kRgaBufferGroupMaxBytes, kRgaBufferGroupLimit) != MPP_OK) {
    mpp_buffer_group_put(group);
    throw std::runtime_error("mpp_buffer_group_limit_config for RGA output failed");
  }
  bufferGroup_ = group;
}

RgbImage RgaPreprocessor::convertAndResize(
    const DecodedFrame& frame,
    int outputWidth,
    int outputHeight,
    const PreprocessOptions& options) {
  if (frame.dmaFd < 0) {
    throw std::runtime_error("Decoded frame does not provide a valid dma fd");
  }
  if (frame.format != PixelFormat::kUnknown && frame.format != PixelFormat::kNv12) {
    throw std::runtime_error("RGA preprocessor currently expects NV12 decoded frames");
  }

  RgbImage output;
  output.width = outputWidth;
  output.height = outputHeight;
  output.format = PixelFormat::kRgb888;
  output.letterbox = buildLetterboxInfo(frame.width, frame.height, outputWidth, outputHeight, options.letterbox);

  const int resizedWidth = output.letterbox.enabled ? output.letterbox.resizedWidth : outputWidth;
  const int resizedHeight = output.letterbox.enabled ? output.letterbox.resizedHeight : outputHeight;
  const bool needsResize = frame.width != resizedWidth || frame.height != resizedHeight;

  // Aligned strides that RGA will see via wrapbuffer_handle.
  const int resizedNv12Wstride = alignUp(resizedWidth, kRgaStrideAlign);
  const int resizedNv12Hstride = alignUp(resizedHeight, 2);
  const int resizedRgbWstride = alignUp(resizedWidth, kRgaStrideAlign);
  const int resizedRgbHstride = resizedHeight;
  const int outputRgbWstride = alignUp(outputWidth, kRgaStrideAlign);
  const int outputRgbHstride = outputHeight;

  const std::size_t resizedNv12Bytes =
      needsResize ? static_cast<std::size_t>(nv12BytesForStride(resizedNv12Wstride, resizedNv12Hstride)) : 0;
  const bool letterboxEnabled = output.letterbox.enabled;
  const std::size_t outputRgbBytes =
      static_cast<std::size_t>(rgb888BytesForStride(outputRgbWstride, outputRgbHstride));
  const std::size_t resizedRgbBytes =
      static_cast<std::size_t>(rgb888BytesForStride(resizedRgbWstride, resizedRgbHstride));
  ensureBufferGroup();

  rga_buffer_handle_t srcHandle = 0;
  rga_buffer_handle_t resizedNv12Handle = 0;
  rga_buffer_handle_t resizedRgbHandle = 0;
  rga_buffer_handle_t outputHandle = 0;
  MppBuffer resizedNv12Buffer = nullptr;
  MppBuffer resizedRgbBuffer = nullptr;
  MppBuffer outputBuffer = nullptr;

  try {
    const im_rect emptyRect = {};
    const rga_buffer_t emptyPat = {};
    const auto group = static_cast<MppBufferGroup>(bufferGroup_);

    outputBuffer = allocateBuffer(group, outputRgbBytes, "RGA output");

    output.dmaFd = mpp_buffer_get_fd(outputBuffer);
    if (output.dmaFd < 0) {
      throw std::runtime_error("mpp_buffer_get_fd failed for RGA output buffer");
    }
    output.wstride = outputRgbWstride;
    output.hstride = outputRgbHstride;
    output.dmaSize = outputRgbBytes;
    output.nativeHandle = makeMppBufferHandle(outputBuffer);
    outputBuffer = nullptr;

    outputHandle = importbuffer_fd(output.dmaFd, static_cast<int>(outputRgbBytes));
    if (outputHandle == 0) {
      throw std::runtime_error("Failed to import RGA output DRM buffer");
    }

    rga_buffer_t outputRgb = wrapbuffer_handle(
        outputHandle,
        outputWidth,
        outputHeight,
        RK_FORMAT_RGB_888,
        outputRgbWstride,
        outputRgbHstride);

    srcHandle = importbuffer_fd(
        frame.dmaFd,
        nv12BytesForStride(frame.horizontalStride, frame.verticalStride));
    if (srcHandle == 0) {
      throw std::runtime_error("Failed to import source RGA buffer (fd)");
    }

    rga_buffer_t src = wrapbuffer_handle(
        srcHandle,
        frame.width,
        frame.height,
        RK_FORMAT_YCbCr_420_SP,
        frame.horizontalStride,
        frame.verticalStride);

    if (needsResize) {
      resizedNv12Buffer = allocateBuffer(group, resizedNv12Bytes, "RGA resized NV12");
      const int resizedNv12Fd = mpp_buffer_get_fd(resizedNv12Buffer);
      if (resizedNv12Fd < 0) {
        throw std::runtime_error("mpp_buffer_get_fd failed for resized NV12 buffer");
      }
      resizedNv12Handle = importbuffer_fd(
          resizedNv12Fd,
          static_cast<int>(resizedNv12Bytes));
      if (resizedNv12Handle == 0) {
        throw std::runtime_error("Failed to import resized NV12 RGA buffer");
      }

      rga_buffer_t resizedNv12 = wrapbuffer_handle(
          resizedNv12Handle,
          resizedWidth,
          resizedHeight,
          RK_FORMAT_YCbCr_420_SP,
          resizedNv12Wstride,
          resizedNv12Hstride);

      checkRgaVerify(imcheck_t(src, resizedNv12, emptyPat, emptyRect, emptyRect, emptyRect, 0), "imresize(NV12)");
      checkRgaOp(imresize(src, resizedNv12), "imresize(NV12)");

      if (letterboxEnabled) {
        resizedRgbBuffer = allocateBuffer(group, resizedRgbBytes, "RGA resized RGB");
        const int resizedRgbFd = mpp_buffer_get_fd(resizedRgbBuffer);
        if (resizedRgbFd < 0) {
          throw std::runtime_error("mpp_buffer_get_fd failed for resized RGB buffer");
        }
        resizedRgbHandle = importbuffer_fd(
            resizedRgbFd,
            static_cast<int>(resizedRgbBytes));
        if (resizedRgbHandle == 0) {
          throw std::runtime_error("Failed to import resized RGB RGA buffer");
        }

        rga_buffer_t resizedRgb = wrapbuffer_handle(
            resizedRgbHandle,
            resizedWidth,
            resizedHeight,
            RK_FORMAT_RGB_888,
            resizedRgbWstride,
            resizedRgbHstride);

        checkRgaVerify(
            imcheck_t(resizedNv12, resizedRgb, emptyPat, emptyRect, emptyRect, emptyRect, 0),
            "imcvtcolor(NV12->RGB)");
        checkRgaOp(
            imcvtcolor(resizedNv12, resizedRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
            "imcvtcolor(NV12->RGB)");

        const int top = output.letterbox.padTop;
        const int bottom = output.letterbox.padBottom;
        const int left = output.letterbox.padLeft;
        const int right = output.letterbox.padRight;

        bool usedHardwareLetterbox = false;
        const int borderCheck = imcheck_t(resizedRgb, outputRgb, emptyPat, emptyRect, emptyRect, emptyRect, 0);
        if (borderCheck == IM_STATUS_NOERROR) {
          const IM_STATUS borderStatus = immakeBorder(
              resizedRgb,
              outputRgb,
              top,
              bottom,
              left,
              right,
              IM_BORDER_CONSTANT,
              static_cast<int>(options.paddingValue),
              1,
              -1,
              nullptr);
          if (borderStatus == IM_STATUS_SUCCESS) {
            usedHardwareLetterbox = true;
          }
        }

        if (!usedHardwareLetterbox) {
          void* outputPtr = mpp_buffer_get_ptr(static_cast<MppBuffer>(output.nativeHandle.get()));
          if (outputPtr == nullptr) {
            throw std::runtime_error("Failed to map RGA output DRM buffer for CPU letterbox fallback");
          }
          const auto* resizedBase =
              static_cast<const std::uint8_t*>(mpp_buffer_get_ptr(resizedRgbBuffer));
          if (resizedBase == nullptr) {
            throw std::runtime_error("Failed to map resized RGB DRM buffer");
          }
          std::memset(outputPtr, options.paddingValue, outputRgbBytes);
          auto* packedOutput = static_cast<std::uint8_t*>(outputPtr);
          for (int y = 0; y < resizedHeight; ++y) {
            std::memcpy(
                packedOutput + static_cast<std::size_t>(y + top) * static_cast<std::size_t>(outputRgbWstride * 3) +
                    static_cast<std::size_t>(left * 3),
                resizedBase + static_cast<std::size_t>(y) * static_cast<std::size_t>(resizedRgbWstride * 3),
                static_cast<std::size_t>(resizedWidth * 3));
          }
        }
      } else {
        checkRgaVerify(
            imcheck_t(resizedNv12, outputRgb, emptyPat, emptyRect, emptyRect, emptyRect, 0),
            "imcvtcolor(NV12->RGB resized direct)");
        checkRgaOp(
            imcvtcolor(resizedNv12, outputRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
            "imcvtcolor(NV12->RGB resized direct)");
      }
    } else {
      checkRgaVerify(
          imcheck_t(src, outputRgb, emptyPat, emptyRect, emptyRect, emptyRect, 0),
          "imcvtcolor(NV12->RGB direct)");
      checkRgaOp(
          imcvtcolor(src, outputRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
          "imcvtcolor(NV12->RGB direct)");
    }

    releaseHandle(srcHandle);
    releaseHandle(resizedNv12Handle);
    releaseHandle(resizedRgbHandle);
    releaseHandle(outputHandle);
    if (resizedNv12Buffer != nullptr) {
      mpp_buffer_put(resizedNv12Buffer);
      resizedNv12Buffer = nullptr;
    }
    if (resizedRgbBuffer != nullptr) {
      mpp_buffer_put(resizedRgbBuffer);
      resizedRgbBuffer = nullptr;
    }
  } catch (...) {
    releaseHandle(srcHandle);
    releaseHandle(resizedNv12Handle);
    releaseHandle(resizedRgbHandle);
    releaseHandle(outputHandle);
    output.nativeHandle.reset();
    output.dmaFd = -1;
    output.wstride = 0;
    output.hstride = 0;
    output.dmaSize = 0;
    if (outputBuffer != nullptr) mpp_buffer_put(outputBuffer);
    if (resizedNv12Buffer != nullptr) mpp_buffer_put(resizedNv12Buffer);
    if (resizedRgbBuffer != nullptr) mpp_buffer_put(resizedRgbBuffer);
    throw;
  }

  if (options.needsCpuData) {
    fillPackedRgbData(output.nativeHandle, outputRgbWstride, outputWidth, outputHeight, output.data);
  }

  return output;
}
