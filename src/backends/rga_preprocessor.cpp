#include "backends/rga_preprocessor.hpp"

extern "C" {
#include <im2d.h>
#include <im2d_buffer.h>
}

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

// RGA hardware requires aligned stride for correct operation. 16 is a safe
// universal choice covering RGA2/RGA3 and both NV12 (min 2) and RGB_888 (min 4).
constexpr int kRgaStrideAlign = 16;

inline int alignUp(int value, int alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// imresize / imcvtcolor return IM_STATUS_SUCCESS (=1) on success.
void checkRgaOp(IM_STATUS status, const char* stage) {
  if (status != IM_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string("RGA ") + stage + " failed: " + imStrError(status));
  }
}

// imcheck returns IM_STATUS_NOERROR (=2) when parameters are valid.
void checkRgaVerify(int ret, const char* stage) {
  if (ret != IM_STATUS_NOERROR) {
    throw std::runtime_error(
        std::string("RGA imcheck before ") + stage + " failed: " +
        imStrError(static_cast<IM_STATUS>(ret)));
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

void blitIntoLetterboxedOutput(
    const std::uint8_t* resized,
    int resizedRowBytes,
    int resizedWidth,
    int resizedHeight,
    std::uint8_t paddingValue,
    const LetterboxInfo& info,
    std::vector<std::uint8_t>& output,
    int outputWidth) {
  std::fill(output.begin(), output.end(), paddingValue);
  const std::size_t copyBytes = static_cast<std::size_t>(resizedWidth * 3);
  const std::size_t dstStride = static_cast<std::size_t>(outputWidth * 3);
  for (int y = 0; y < resizedHeight; ++y) {
    std::memcpy(
        output.data() + static_cast<std::size_t>(y + info.padTop) * dstStride +
            static_cast<std::size_t>(info.padLeft * 3),
        resized + static_cast<std::size_t>(y) * resizedRowBytes,
        copyBytes);
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

}  // namespace

void RgaPreprocessor::ensureWorkspace(
    std::size_t resizedNv12Bytes,
    std::size_t resizedRgbBytes,
    std::size_t outputBytes) {
  if (resizedNv12_.size() < resizedNv12Bytes) {
    resizedNv12_.resize(resizedNv12Bytes);
  }
  if (resizedRgb_.size() < resizedRgbBytes) {
    resizedRgb_.resize(resizedRgbBytes);
  }
  if (outputRgb_.size() < outputBytes) {
    outputRgb_.resize(outputBytes);
  }
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
  const std::size_t resizedRgbBytes =
      static_cast<std::size_t>(rgb888BytesForStride(resizedRgbWstride, resizedRgbHstride));
  // outputRgb_ is written by RGA when letterbox is disabled; reserve space for its aligned stride.
  const bool letterboxEnabled = output.letterbox.enabled;
  const std::size_t outputRgbWorkspaceBytes = letterboxEnabled
      ? static_cast<std::size_t>(outputWidth * outputHeight * 3)  // packed, for CPU letterbox blit
      : static_cast<std::size_t>(rgb888BytesForStride(outputRgbWstride, outputRgbHstride));
  ensureWorkspace(resizedNv12Bytes, resizedRgbBytes, outputRgbWorkspaceBytes);

  rga_buffer_handle_t srcHandle = 0;
  rga_buffer_handle_t resizedNv12Handle = 0;
  rga_buffer_handle_t rgbHandle = 0;

  try {
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
      resizedNv12Handle = importbuffer_virtualaddr(
          resizedNv12_.data(),
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

      checkRgaVerify(imcheck(src, resizedNv12, {}, {}), "imresize(NV12)");
      checkRgaOp(imresize(src, resizedNv12), "imresize(NV12)");

      rgbHandle = importbuffer_virtualaddr(
          resizedRgb_.data(),
          static_cast<int>(resizedRgbBytes));
      if (rgbHandle == 0) {
        throw std::runtime_error("Failed to import resized RGB RGA buffer");
      }

      rga_buffer_t resizedRgb = wrapbuffer_handle(
          rgbHandle,
          resizedWidth,
          resizedHeight,
          RK_FORMAT_RGB_888,
          resizedRgbWstride,
          resizedRgbHstride);

      checkRgaVerify(imcheck(resizedNv12, resizedRgb, {}, {}), "imcvtcolor(NV12->RGB)");
      checkRgaOp(
          imcvtcolor(resizedNv12, resizedRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
          "imcvtcolor(NV12->RGB)");
    } else {
      rgbHandle = importbuffer_virtualaddr(
          outputRgb_.data(),
          static_cast<int>(outputRgbWorkspaceBytes));
      if (rgbHandle == 0) {
        throw std::runtime_error("Failed to import output RGB RGA buffer");
      }

      rga_buffer_t outputRgb = wrapbuffer_handle(
          rgbHandle,
          resizedWidth,
          resizedHeight,
          RK_FORMAT_RGB_888,
          outputRgbWstride,
          outputRgbHstride);

      checkRgaVerify(imcheck(src, outputRgb, {}, {}), "imcvtcolor(NV12->RGB direct)");
      checkRgaOp(
          imcvtcolor(src, outputRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
          "imcvtcolor(NV12->RGB direct)");
    }

    releaseHandle(srcHandle);
    releaseHandle(resizedNv12Handle);
    releaseHandle(rgbHandle);
  } catch (...) {
    releaseHandle(srcHandle);
    releaseHandle(resizedNv12Handle);
    releaseHandle(rgbHandle);
    throw;
  }

  const std::size_t packedOutputBytes = static_cast<std::size_t>(outputWidth * outputHeight * 3);
  output.data.resize(packedOutputBytes);

  if (letterboxEnabled) {
    blitIntoLetterboxedOutput(
        resizedRgb_.data(),
        resizedRgbWstride * 3,
        resizedWidth,
        resizedHeight,
        options.paddingValue,
        output.letterbox,
        output.data,
        outputWidth);
  } else {
    // RGA wrote into outputRgb_ using outputRgbWstride; pack into tight output.data.
    copyStridedToPacked(
        outputRgb_.data(),
        outputRgbWstride * 3,
        output.data.data(),
        outputWidth * 3,
        outputHeight);
  }

  return output;
}
