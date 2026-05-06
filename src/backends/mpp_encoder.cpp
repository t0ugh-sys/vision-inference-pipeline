#include "backends/mpp_encoder.hpp"
#include "rga_shared.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <mpp_buffer.h>
#include <mpp_err.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
#include <rk_mpi.h>
#include <rk_mpi_cmd.h>
}

#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
#include <im2d.h>
#include <im2d_buffer.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <stdexcept>

namespace {

constexpr size_t kPacketBufferMinSize = 512 * 1024;
constexpr auto kDrainRetrySleep = std::chrono::milliseconds(2);
constexpr int kDrainRetryCount = 50;
constexpr int kFlushDrainRetryCount = 5000;
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

void checkMppStatus(MPP_RET status, const char* message) {
  if (status != MPP_OK) {
    throw std::runtime_error(message);
  }
}

#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
void checkRgaStatus(IM_STATUS status, const char* message) {
  if (status != IM_STATUS_SUCCESS) {
    throw std::runtime_error(message);
  }
}
#endif

MppCodingType toCodingType(const std::string& codec) {
  if (codec == "hevc" || codec == "h265") {
    return MPP_VIDEO_CodingHEVC;
  }
  return MPP_VIDEO_CodingAVC;
}

size_t packetBufferSize(const EncoderConfig& config) {
  const size_t frameBytes = static_cast<size_t>(std::max(config.horStride, config.width)) *
                            static_cast<size_t>(std::max(config.verStride, config.height)) * 3 / 2;
  return std::max(frameBytes, kPacketBufferMinSize);
}

int alignUp(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

size_t alignedFrameBytes(int horStride, int verStride, PixelFormat format) {
  const size_t alignedHor = static_cast<size_t>(alignUp(horStride, 64));
  const size_t alignedVer = static_cast<size_t>(alignUp(verStride, 64));
  if (format == PixelFormat::kRgb888) {
    return alignedHor * alignedVer;
  }
  return alignedHor * alignedVer * 3 / 2;
}

bool verboseMppRgbEncodeLogsEnabled() {
  const char* value = std::getenv("MPP_RGB_ENCODER_VERBOSE_LOG");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool useNv21ForRgbEncode() {
  const char* value = std::getenv("MPP_ENCODER_USE_NV21");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool lowLatencyEncodeRequested(bool rtspOutput, int explicitMode) {
  if (explicitMode >= 0) {
    return explicitMode != 0;
  }
  if (rtspOutput) {
    return true;
  }
  const char* value = std::getenv("MPP_ENCODER_LOW_LATENCY");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool mppEncoderTimingEnabled() {
  const char* value = std::getenv("MPP_ENCODER_TIMING");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

int qualityQpMaxForBitrate(int targetBitrate) {
  if (targetBitrate >= 16'000'000) {
    return 28;
  }
  if (targetBitrate >= 8'000'000) {
    return 34;
  }
  return 45;
}

void logMppRgbEncodeStep(const char* step) {
  if (!verboseMppRgbEncodeLogsEnabled()) {
    return;
  }
  std::cerr << "[MPP-RGB] " << step << "\n";
  std::cerr.flush();
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

bool isIgnorableRtspWriteError(int status) {
  return status == AVERROR_EOF ||
         status == AVERROR(EPIPE) ||
         status == AVERROR(ECONNRESET) ||
         status == AVERROR(ECONNABORTED) ||
         status == AVERROR(ETIMEDOUT);
}

int64_t computePacketDurationTicks(int fpsNum, int fpsDen, const AVRational& timeBase) {
  const int safeFpsNum = std::max(1, fpsNum);
  const int safeFpsDen = std::max(1, fpsDen);
  const int64_t ticks = av_rescale_q(
      1,
      AVRational{safeFpsDen, safeFpsNum},
      timeBase);
  return std::max<int64_t>(1, ticks);
}

AVCodecID toAvCodecId(const std::string& codec) {
  if (codec == "hevc" || codec == "h265") {
    return AV_CODEC_ID_HEVC;
  }
  return AV_CODEC_ID_H264;
}

void checkAvStatus(int status, const char* message) {
  if (status >= 0) {
    return;
  }
  char errorBuffer[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(status, errorBuffer, sizeof(errorBuffer));
  throw std::runtime_error(std::string(message) + ": " + errorBuffer);
}

bool hasAnnexBStartCode(const std::uint8_t* data, std::size_t size, std::size_t offset, std::size_t* startCodeSize) {
  if (offset + 3 <= size &&
      data[offset] == 0x00 &&
      data[offset + 1] == 0x00 &&
      data[offset + 2] == 0x01) {
    *startCodeSize = 3;
    return true;
  }
  if (offset + 4 <= size &&
      data[offset] == 0x00 &&
      data[offset + 1] == 0x00 &&
      data[offset + 2] == 0x00 &&
      data[offset + 3] == 0x01) {
    *startCodeSize = 4;
    return true;
  }
  return false;
}

bool containsH264IdrNal(const std::uint8_t* data, std::size_t size) {
  if (data == nullptr || size < 5) {
    return false;
  }

  for (std::size_t i = 0; i + 4 <= size; ++i) {
    std::size_t startCodeSize = 0;
    if (!hasAnnexBStartCode(data, size, i, &startCodeSize)) {
      continue;
    }

    const std::size_t nalOffset = i + startCodeSize;
    if (nalOffset >= size) {
      break;
    }

    const std::uint8_t nalType = data[nalOffset] & 0x1f;
    if (nalType == 5) {
      return true;
    }
  }

  return false;
}

bool isH264ParameterNalType(std::uint8_t nalType) {
  return nalType == 7 || nalType == 8 || nalType == 9;
}

const std::uint8_t* filterRtspAnnexBPayload(
    const std::uint8_t* data,
    std::size_t size,
    std::vector<std::uint8_t>& scratch,
    std::size_t* filteredSize) {
  if (data == nullptr || size == 0) {
    *filteredSize = 0;
    return nullptr;
  }

  scratch.clear();
  bool removedParameterSets = false;
  bool keptAnyNal = false;

  std::size_t nalStart = 0;
  while (nalStart < size) {
    std::size_t startCodeSize = 0;
    if (!hasAnnexBStartCode(data, size, nalStart, &startCodeSize)) {
      ++nalStart;
      continue;
    }

    const std::size_t nalHeaderOffset = nalStart + startCodeSize;
    if (nalHeaderOffset >= size) {
      break;
    }

    std::size_t nalEnd = nalHeaderOffset;
    while (nalEnd < size) {
      std::size_t nextStartCodeSize = 0;
      if (hasAnnexBStartCode(data, size, nalEnd, &nextStartCodeSize)) {
        break;
      }
      ++nalEnd;
    }

    const std::uint8_t nalType = data[nalHeaderOffset] & 0x1f;
    if (isH264ParameterNalType(nalType)) {
      removedParameterSets = true;
    } else {
      scratch.insert(scratch.end(), data + nalStart, data + nalEnd);
      keptAnyNal = true;
    }

    nalStart = nalEnd;
  }

  if (!removedParameterSets) {
    *filteredSize = size;
    return data;
  }

  if (!keptAnyNal) {
    *filteredSize = 0;
    return nullptr;
  }

  *filteredSize = scratch.size();
  return scratch.data();
}

}  // namespace

MppEncoder::MppEncoder() = default;

MppEncoder::~MppEncoder() {
  close();
}

void MppEncoder::init(const EncoderConfig& config) {
  close();
  logMppRgbEncodeStep("init_begin");

  if (config.width <= 0 || config.height <= 0) {
    throw std::runtime_error("MPP encoder requires a positive frame size");
  }
  if (config.inputFormat != PixelFormat::kNv12 &&
      config.inputFormat != PixelFormat::kRgb888) {
    throw std::runtime_error("MPP encoder only supports NV12 decoded-frame input or RGB888 frames");
  }

  width_ = config.width;
  height_ = config.height;
  horStride_ = config.horStride > 0 ? config.horStride : alignUp(config.width, 8);
  verStride_ = config.verStride > 0 ? config.verStride : alignUp(config.height, 2);
  inputFormat_ = config.inputFormat;
  frameFormat_ = MPP_FMT_YUV420SP;
  rgaYuvFormat_ = RK_FORMAT_YCbCr_420_SP;
  if (inputFormat_ == PixelFormat::kRgb888 && useNv21ForRgbEncode()) {
    frameFormat_ = MPP_FMT_YUV420SP_VU;
    rgaYuvFormat_ = RK_FORMAT_YCrCb_420_SP;
  }
  flushSubmitted_ = false;
  frameIndex_ = 0;
  fpsNum_ = config.fpsNum > 0 ? config.fpsNum : (config.fps > 0 ? config.fps : 30);
  fpsDen_ = config.fpsDen > 0 ? config.fpsDen : 1;
  nextPacketPts_ = 0;
  outputMuxingRequested_ = hasSuffixIgnoreCase(config.outputPath, ".mp4") || isRtspUrl(config.outputPath);
  rtspOutput_ = isRtspUrl(config.outputPath);
  outputDisconnected_ = false;
  timingEnabled_ = mppEncoderTimingEnabled();
  initOutputSink(config);
  logMppRgbEncodeStep("init_output_opened");

  if (inputFormat_ == PixelFormat::kRgb888) {
    checkMppStatus(mpp_buffer_group_get_internal(
                       &bufferGroup_,
                       static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)),
                   "mpp_buffer_group_get_internal failed");
    logMppRgbEncodeStep("init_buffer_group_done");
    const size_t frameBytes = alignedFrameBytes(horStride_, verStride_, PixelFormat::kNv12);
    checkMppStatus(mpp_buffer_group_limit_config(bufferGroup_, frameBytes, 2),
                   "mpp_buffer_group_limit_config failed");
    logMppRgbEncodeStep("init_buffer_limit_done");
    checkMppStatus(mpp_buffer_get(bufferGroup_, &inputBuffer_, frameBytes),
                   "mpp_buffer_get for encoder input failed");
    logMppRgbEncodeStep("init_input_buffer_done");
  } else {
    checkMppStatus(mpp_buffer_group_get_internal(
                       &bufferGroup_,
                       static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)),
                   "mpp_buffer_group_get_internal failed");
    logMppRgbEncodeStep("init_buffer_group_done");
  }

  const size_t packetBytes = std::max(
      alignedFrameBytes(horStride_, verStride_, PixelFormat::kNv12),
      packetBufferSize(config));
  checkMppStatus(mpp_buffer_get(bufferGroup_, &packetBuffer_, packetBytes),
                 "mpp_buffer_get for output packet failed");
  logMppRgbEncodeStep("init_packet_buffer_done");

  checkMppStatus(mpp_create(&context_, &api_), "mpp_create failed");
  logMppRgbEncodeStep("init_mpp_create_done");
  checkMppStatus(mpp_init(context_, MPP_CTX_ENC, toCodingType(config.codec)), "mpp_init failed");
  logMppRgbEncodeStep("init_mpp_init_done");

  MppPollType timeout = MPP_POLL_BLOCK;
  checkMppStatus(api_->control(context_, MPP_SET_OUTPUT_TIMEOUT, &timeout),
                 "MPP_SET_OUTPUT_TIMEOUT failed");
  logMppRgbEncodeStep("init_set_timeout_done");
  const int targetBitrate = config.bitrate > 0 ? config.bitrate : 4000000;
  const int qpMax = qualityQpMaxForBitrate(targetBitrate);
  const bool lowLatencyEncode = lowLatencyEncodeRequested(rtspOutput_, config.lowLatency);

  if (config.codec == "hevc" || config.codec == "h265") {
    throw std::runtime_error("Struct-based Rockchip encoder init currently supports h264 output only");
  }

  checkMppStatus(mpp_enc_cfg_init(&config_), "mpp_enc_cfg_init failed");
  logMppRgbEncodeStep("init_cfg_alloc_done");
  checkMppStatus(api_->control(context_, MPP_ENC_GET_CFG, config_),
                 "MPP_ENC_GET_CFG failed");
  logMppRgbEncodeStep("init_cfg_get_done");

  // Use the official MPP_ENC_SET_CFG path instead of the deprecated
  // SET_PREP_CFG / SET_RC_CFG / SET_CODEC_CFG controls. This matches the
  // supported init flow in mpi_enc_test and avoids board-specific drift.
  mpp_enc_cfg_set_s32(config_, "codec:type", MPP_VIDEO_CodingAVC);
  mpp_enc_cfg_set_s32(config_, "prep:width", width_);
  mpp_enc_cfg_set_s32(config_, "prep:height", height_);
  mpp_enc_cfg_set_s32(config_, "prep:hor_stride", horStride_);
  mpp_enc_cfg_set_s32(config_, "prep:ver_stride", verStride_);
  mpp_enc_cfg_set_s32(config_, "prep:format", frameFormat_);
  mpp_enc_cfg_set_s32(config_, "prep:range", MPP_FRAME_RANGE_JPEG);

  mpp_enc_cfg_set_s32(config_, "rc:mode", MPP_ENC_RC_MODE_CBR);
  mpp_enc_cfg_set_u32(config_, "rc:max_reenc_times", 0);
  mpp_enc_cfg_set_u32(config_, "rc:super_mode", 0);
  mpp_enc_cfg_set_u32(config_, "rc:drop_mode", MPP_ENC_RC_DROP_FRM_DISABLED);
  mpp_enc_cfg_set_u32(config_, "rc:drop_thd", 20);
  mpp_enc_cfg_set_u32(config_, "rc:drop_gap", 1);
  mpp_enc_cfg_set_s32(config_, "rc:fps_in_flex", 0);
  mpp_enc_cfg_set_s32(config_, "rc:fps_in_num", fpsNum_);
  mpp_enc_cfg_set_s32(config_, "rc:fps_in_denom", fpsDen_);
  mpp_enc_cfg_set_s32(config_, "rc:fps_out_flex", 0);
  mpp_enc_cfg_set_s32(config_, "rc:fps_out_num", fpsNum_);
  mpp_enc_cfg_set_s32(config_, "rc:fps_out_denom", fpsDen_);
  mpp_enc_cfg_set_s32(config_, "rc:gop", std::max(1, (fpsNum_ + fpsDen_ - 1) / fpsDen_));
  mpp_enc_cfg_set_s32(config_, "rc:bps_target", targetBitrate);
  mpp_enc_cfg_set_s32(config_, "rc:bps_max", targetBitrate * 17 / 16);
  mpp_enc_cfg_set_s32(config_, "rc:bps_min", std::max(1, targetBitrate * 15 / 16));
  mpp_enc_cfg_set_s32(config_, "rc:qp_init", -1);
  mpp_enc_cfg_set_s32(config_, "rc:qp_max", qpMax);
  mpp_enc_cfg_set_s32(config_, "rc:qp_min", 8);
  mpp_enc_cfg_set_s32(config_, "rc:qp_max_i", qpMax);
  mpp_enc_cfg_set_s32(config_, "rc:qp_min_i", 8);
  mpp_enc_cfg_set_s32(config_, "rc:qp_ip", 2);
  mpp_enc_cfg_set_s32(config_, "rc:fqp_min_i", 8);
  mpp_enc_cfg_set_s32(config_, "rc:fqp_max_i", qpMax);
  mpp_enc_cfg_set_s32(config_, "rc:fqp_min_p", 8);
  mpp_enc_cfg_set_s32(config_, "rc:fqp_max_p", qpMax);

  mpp_enc_cfg_set_s32(config_, "h264:profile", lowLatencyEncode ? 66 : 100);
  mpp_enc_cfg_set_s32(config_, "h264:level", (width_ >= 1920 || height_ >= 1080) ? 40 : 31);
  mpp_enc_cfg_set_s32(config_, "h264:cabac_en", lowLatencyEncode ? 0 : 1);
  mpp_enc_cfg_set_s32(config_, "h264:cabac_idc", 0);
  mpp_enc_cfg_set_s32(config_, "h264:trans8x8", lowLatencyEncode ? 0 : 1);

  if (rtspOutput_) {
    // Keep RTSP-oriented output easier to packetize and decode in real time.
    mpp_enc_cfg_set_s32(config_, "rc:gop", std::max(1, (fpsNum_ + fpsDen_ - 1) / fpsDen_));
  }

  checkMppStatus(api_->control(context_, MPP_ENC_SET_CFG, config_),
                 "MPP_ENC_SET_CFG failed");
  logMppRgbEncodeStep("init_cfg_set_done");

  MppPacket headerPacket = nullptr;
  checkMppStatus(mpp_packet_init_with_buffer(&headerPacket, packetBuffer_), "mpp_packet_init_with_buffer failed");
  logMppRgbEncodeStep("init_header_packet_done");
  mpp_packet_set_length(headerPacket, 0);
  checkMppStatus(api_->control(context_, MPP_ENC_GET_HDR_SYNC, headerPacket), "MPP_ENC_GET_HDR_SYNC failed");
  logMppRgbEncodeStep("init_get_hdr_done");
  if (outputMuxingRequested_) {
    void* headerData = mpp_packet_get_pos(headerPacket);
    const size_t headerLength = mpp_packet_get_length(headerPacket);
    if (headerData == nullptr || headerLength == 0) {
      throw std::runtime_error("MPP_ENC_GET_HDR_SYNC returned empty codec header");
    }
    muxHeader_.assign(
        static_cast<const std::uint8_t*>(headerData),
        static_cast<const std::uint8_t*>(headerData) + headerLength);
    initMp4Muxer(config);
  } else {
    // Write SPS/PPS once during init so the raw .h264 output can be remuxed
    // later without depending on an out-of-band header source.
    writePacket(headerPacket);
  }
  mpp_packet_deinit(&headerPacket);
  logMppRgbEncodeStep("init_done");

  initialized_ = true;
}

void MppEncoder::encode(const RgbImage& frame, int64_t pts) {
#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
  logMppRgbEncodeStep("encode_begin");
  if (!initialized_) {
    throw std::runtime_error("MPP encoder is not initialized");
  }
  if (outputDisconnected_) {
    return;
  }
  if (inputFormat_ != PixelFormat::kRgb888) {
    throw std::runtime_error("MPP encoder is not configured for RGB input");
  }
  if (frame.width != width_ || frame.height != height_) {
    throw std::runtime_error("MPP encoder RGB input size mismatch");
  }
  if (frame.data.size() != static_cast<std::size_t>(frame.width * frame.height * 3)) {
    throw std::runtime_error("MPP encoder received an invalid RGB frame buffer");
  }
  if (inputBuffer_ == nullptr) {
    throw std::runtime_error("MPP encoder RGB input buffer is not allocated");
  }

  logMppRgbEncodeStep("got_input_buffer");
  {
    std::lock_guard<std::mutex> lock(globalRgaMutex());
    // Let RGA write directly into the MPP-owned dma-buf. Earlier host memset /
    // virtual-address staging paths were a source of visible line artifacts.
    rga_buffer_handle_t srcHandle =
        importbuffer_virtualaddr(const_cast<std::uint8_t*>(frame.data.data()), frame.data.size());
    if (srcHandle == 0) {
      throw std::runtime_error("RGA importbuffer_virtualaddr failed for RGB encoder source");
    }
    const int inputFd = mpp_buffer_get_fd(inputBuffer_);
    if (inputFd < 0) {
      releasebuffer_handle(srcHandle);
      throw std::runtime_error("mpp_buffer_get_fd failed for encoder NV12 destination");
    }
    rga_buffer_handle_t dstHandle = importbuffer_fd(inputFd, horStride_, verStride_, rgaYuvFormat_);
    if (dstHandle == 0) {
      releasebuffer_handle(srcHandle);
      throw std::runtime_error("RGA importbuffer_fd failed for NV12 encoder destination");
    }

    try {
      rga_buffer_t src = wrapbuffer_handle(srcHandle, width_, height_, RK_FORMAT_RGB_888);
      rga_buffer_t dst = wrapbuffer_handle(
          dstHandle,
          width_,
          height_,
          rgaYuvFormat_,
          horStride_,
          verStride_);
      checkRgaStatus(
          imcvtcolor(src, dst, RK_FORMAT_RGB_888, rgaYuvFormat_),
          "RGA RGB to YUV420 conversion failed");
      releasebuffer_handle(dstHandle);
      releasebuffer_handle(srcHandle);
    } catch (...) {
      releasebuffer_handle(dstHandle);
      releasebuffer_handle(srcHandle);
      throw;
    }
  }
  logMppRgbEncodeStep("rgb_to_yuv_done");

  MppFrame inputFrame = nullptr;
  try {
    if (frameIndex_ == 0) {
      // Force the first encoded frame to be an IDR. Without this, the raw stream
      // can start with an unusable P-frame and downstream remux/decode becomes fragile.
      checkMppStatus(api_->control(context_, MPP_ENC_SET_IDR_FRAME, nullptr),
                     "MPP_ENC_SET_IDR_FRAME failed");
    }
    checkMppStatus(mpp_frame_init(&inputFrame), "mpp_frame_init failed");
    logMppRgbEncodeStep("frame_init_done");
    mpp_frame_set_width(inputFrame, width_);
    mpp_frame_set_height(inputFrame, height_);
    mpp_frame_set_hor_stride(inputFrame, horStride_);
    mpp_frame_set_ver_stride(inputFrame, verStride_);
    mpp_frame_set_fmt(inputFrame, frameFormat_);
    // The pipeline PTS comes from the source demuxer time base, which is not
    // available here. Feed MPP a monotonic frame index so downstream muxing
    // uses a stable encoder-owned timeline for MP4/RTSP output.
    mpp_frame_set_pts(inputFrame, frameIndex_);
    ++frameIndex_;
    mpp_frame_set_buffer(inputFrame, inputBuffer_);

    checkMppStatus(api_->encode_put_frame(context_, inputFrame), "encode_put_frame failed");
    logMppRgbEncodeStep("encode_put_frame_done");
    mpp_frame_deinit(&inputFrame);
    bool gotPacket = false;
    int retries = 0;
    while (true) {
      MppPacket encodedPacket = nullptr;
      checkMppStatus(api_->encode_get_packet(context_, &encodedPacket), "encode_get_packet failed");
      if (encodedPacket == nullptr) {
        // MPP can accept a frame before its output packet is immediately ready.
        // Give the encoder a short bounded retry window instead of treating
        // the first null packet as hard failure / truncated frame.
        if (!gotPacket && retries++ < kDrainRetryCount) {
          std::this_thread::sleep_for(kDrainRetrySleep);
          continue;
        }
        break;
      }
      gotPacket = true;
      retries = 0;
      logMppRgbEncodeStep("encode_get_packet_done");
      writePacket(encodedPacket);
      mpp_packet_deinit(&encodedPacket);
      break;
    }
    logMppRgbEncodeStep("encode_done");
  } catch (...) {
    if (inputFrame != nullptr) {
      mpp_frame_deinit(&inputFrame);
    }
    throw;
  }
#else
  (void)frame;
  (void)pts;
  throw std::runtime_error("MPP encoder RGB input requires Rockchip RGA support");
#endif
}

void MppEncoder::encodeDecodedFrame(const DecodedFrame& frame, int64_t pts) {
  if (!initialized_) {
    throw std::runtime_error("MPP encoder is not initialized");
  }
  if (outputDisconnected_) {
    return;
  }
  if (frame.dmaFd < 0) {
    throw std::runtime_error("MPP encoder requires a valid dma-buf fd");
  }
  if (frame.format != PixelFormat::kNv12 && frame.format != PixelFormat::kUnknown) {
    throw std::runtime_error("MPP encoder currently only supports NV12 decoded frames");
  }
  if (frame.width != width_ || frame.height != height_) {
    throw std::runtime_error("MPP encoder does not support resolution changes yet");
  }

  MppBuffer inputBuffer = nullptr;
  MppFrame inputFrame = nullptr;

  try {
    MppBufferInfo bufferInfo;
    std::memset(&bufferInfo, 0, sizeof(bufferInfo));
    bufferInfo.type = MPP_BUFFER_TYPE_EXT_DMA;
    bufferInfo.fd = frame.dmaFd;
    bufferInfo.size = static_cast<size_t>(frame.horizontalStride) * static_cast<size_t>(frame.verticalStride) * 3 / 2;
    checkMppStatus(mpp_buffer_import(&inputBuffer, &bufferInfo), "mpp_buffer_import failed");

    if (frameIndex_ == 0) {
      checkMppStatus(api_->control(context_, MPP_ENC_SET_IDR_FRAME, nullptr),
                     "MPP_ENC_SET_IDR_FRAME failed");
    }
    checkMppStatus(mpp_frame_init(&inputFrame), "mpp_frame_init failed");
    mpp_frame_set_width(inputFrame, frame.width);
    mpp_frame_set_height(inputFrame, frame.height);
    mpp_frame_set_hor_stride(inputFrame, frame.horizontalStride > 0 ? frame.horizontalStride : horStride_);
    mpp_frame_set_ver_stride(inputFrame, frame.verticalStride > 0 ? frame.verticalStride : verStride_);
    mpp_frame_set_fmt(inputFrame, MPP_FMT_YUV420SP);
    // Decoded-frame PTS values are still expressed in the source stream's time
    // base. The MPP encoder/mux path owns a fixed FPS timeline, so use the
    // encoded frame index instead of forwarding the demuxer time base directly.
    mpp_frame_set_pts(inputFrame, frameIndex_);
    mpp_frame_set_buffer(inputFrame, inputBuffer);

    const auto putStart = Clock::now();
    checkMppStatus(api_->encode_put_frame(context_, inputFrame), "encode_put_frame failed");
    totalEncodePutMs_ += Ms(Clock::now() - putStart).count();
    ++frameIndex_;
    mpp_frame_deinit(&inputFrame);
    mpp_buffer_put(inputBuffer);
    inputBuffer = nullptr;
    bool gotPacket = false;
    int retries = 0;
    const auto getStart = Clock::now();
    while (true) {
      MppPacket packet = nullptr;
      checkMppStatus(api_->encode_get_packet(context_, &packet), "encode_get_packet failed");
      if (packet == nullptr) {
        if (!gotPacket && retries++ < kDrainRetryCount) {
          std::this_thread::sleep_for(kDrainRetrySleep);
          continue;
        }
        break;
      }
      gotPacket = true;
      retries = 0;
      writePacket(packet);
      mpp_packet_deinit(&packet);
      break;
    }
    totalEncodeGetMs_ += Ms(Clock::now() - getStart).count();
    ++timedFrameCount_;
    if (timingEnabled_ && (timedFrameCount_ == 1 || (timedFrameCount_ % 300 == 0))) {
      std::cerr << "[MPP] timing"
                << " frames=" << timedFrameCount_
                << " put_ms_avg=" << (totalEncodePutMs_ / timedFrameCount_)
                << " get_ms_avg=" << (totalEncodeGetMs_ / timedFrameCount_)
                << " write_ms_avg=" << (totalWritePacketMs_ / timedFrameCount_)
                << "\n";
    }
  } catch (...) {
    if (inputFrame != nullptr) {
      mpp_frame_deinit(&inputFrame);
    }
    if (inputBuffer != nullptr) {
      mpp_buffer_put(inputBuffer);
    }
    throw;
  }
}

void MppEncoder::flush() {
  if (!initialized_ || flushSubmitted_) {
    return;
  }
  if (outputDisconnected_) {
    flushSubmitted_ = true;
    return;
  }

  MppFrame eosFrame = nullptr;
  try {
    checkMppStatus(mpp_frame_init(&eosFrame), "mpp_frame_init failed");
    mpp_frame_set_eos(eosFrame, 1);
    checkMppStatus(api_->encode_put_frame(context_, eosFrame), "encode_put_frame(eos) failed");
    flushSubmitted_ = true;
    mpp_frame_deinit(&eosFrame);
    drainPackets(true);
    if (outputFile_.is_open()) {
      outputFile_.flush();
    }
  } catch (...) {
    if (eosFrame != nullptr) {
      mpp_frame_deinit(&eosFrame);
    }
    throw;
  }
}

void MppEncoder::close() {
  closeOutputSink();
  if (inputBuffer_ != nullptr) {
    mpp_buffer_put(inputBuffer_);
    inputBuffer_ = nullptr;
  }
  if (packetBuffer_ != nullptr) {
    mpp_buffer_put(packetBuffer_);
    packetBuffer_ = nullptr;
  }
  if (bufferGroup_ != nullptr) {
    mpp_buffer_group_put(bufferGroup_);
    bufferGroup_ = nullptr;
  }
  if (config_ != nullptr) {
    mpp_enc_cfg_deinit(config_);
    config_ = nullptr;
  }
  if (context_ != nullptr) {
    mpp_destroy(context_);
    context_ = nullptr;
    api_ = nullptr;
  }

  initialized_ = false;
  flushSubmitted_ = false;
  outputMuxingRequested_ = false;
  muxHeaderWritten_ = false;
  rtspOutput_ = false;
  outputDisconnected_ = false;
  width_ = 0;
  height_ = 0;
  horStride_ = 0;
  verStride_ = 0;
  frameIndex_ = 0;
  inputFormat_ = PixelFormat::kUnknown;
  fpsNum_ = 30;
  fpsDen_ = 1;
  nextPacketPts_ = 0;
  packetDurationTicks_ = 1;
  packetTimebaseNum_ = 1;
  packetTimebaseDen_ = 1;
}

void MppEncoder::writePacket(void* opaquePacket) {
  const auto writeStart = Clock::now();
  MppPacket packet = static_cast<MppPacket>(opaquePacket);
  if (!packet) {
    return;
  }
  if (outputDisconnected_) {
    return;
  }

  void* data = mpp_packet_get_pos(packet);
  size_t length = mpp_packet_get_length(packet);
  if (data == nullptr || length == 0) {
    return;
  }

  if (writesMp4Container()) {
    if (rtspOutput_) {
      data = const_cast<std::uint8_t*>(filterRtspAnnexBPayload(
          static_cast<const std::uint8_t*>(data),
          length,
          muxPacketScratch_,
          &length));
      if (data == nullptr || length == 0) {
        return;
      }
    }
    // MPP gives Annex-B H.264. Mark IDR access units as keyframes so MP4 seeking stays sane.
    const bool isKeyframe = containsH264IdrNal(static_cast<const std::uint8_t*>(data), length);
    AVPacket muxPacket;
    av_init_packet(&muxPacket);
    muxPacket.data = static_cast<std::uint8_t*>(data);
    muxPacket.size = static_cast<int>(length);
    muxPacket.stream_index = muxVideoStream_->index;
    muxPacket.pts = nextPacketPts_;
    muxPacket.dts = nextPacketPts_;
    muxPacket.duration = packetDurationTicks_;
    muxPacket.flags = isKeyframe ? AV_PKT_FLAG_KEY : 0;
    av_packet_rescale_ts(
        &muxPacket,
        AVRational{packetTimebaseNum_, packetTimebaseDen_},
        muxVideoStream_->time_base);
    const int writeStatus = av_interleaved_write_frame(muxFormatContext_, &muxPacket);
    if (rtspOutput_ && isIgnorableRtspWriteError(writeStatus)) {
      outputDisconnected_ = true;
      std::cerr << "[WARN] RTSP output disconnected, stopping packet writes\n";
      std::cerr.flush();
      return;
    }
    checkAvStatus(writeStatus, "av_interleaved_write_frame failed");
    nextPacketPts_ += packetDurationTicks_;
    totalWritePacketMs_ += Ms(Clock::now() - writeStart).count();
    return;
  }

  outputFile_.write(static_cast<const char*>(data), static_cast<std::streamsize>(length));
  totalWritePacketMs_ += Ms(Clock::now() - writeStart).count();
}

void MppEncoder::initOutputSink(const EncoderConfig& config) {
  closeOutputSink();
  if (outputMuxingRequested_) {
    return;
  }
  outputFile_ = std::ofstream(config.outputPath, std::ios::binary);
  if (!outputFile_.is_open()) {
    throw std::runtime_error("Failed to open output file: " + config.outputPath);
  }
}

void MppEncoder::initMp4Muxer(const EncoderConfig& config) {
  avformat_network_init();
  const bool outputRtsp = isRtspUrl(config.outputPath);
  const char* formatName = outputRtsp ? "rtsp" : "mp4";
  checkAvStatus(
      avformat_alloc_output_context2(&muxFormatContext_, nullptr, formatName, config.outputPath.c_str()),
      "avformat_alloc_output_context2 failed");
  if (muxFormatContext_ == nullptr) {
    throw std::runtime_error(std::string("Failed to create ") + (outputRtsp ? "RTSP" : "MP4") + " output context");
  }

  muxVideoStream_ = avformat_new_stream(muxFormatContext_, nullptr);
  if (muxVideoStream_ == nullptr) {
    throw std::runtime_error("avformat_new_stream failed");
  }

  if (outputRtsp) {
    muxVideoStream_->time_base = AVRational{1, 90000};
  } else {
    muxVideoStream_->time_base = AVRational{fpsDen_, fpsNum_};
  }
  muxVideoStream_->avg_frame_rate = AVRational{fpsNum_, fpsDen_};
  muxVideoStream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
  muxVideoStream_->codecpar->codec_id = toAvCodecId(config.codec);
  muxVideoStream_->codecpar->width = width_;
  muxVideoStream_->codecpar->height = height_;
  muxVideoStream_->codecpar->format = AV_PIX_FMT_YUV420P;

  if (!muxHeader_.empty()) {
    muxVideoStream_->codecpar->extradata =
        static_cast<std::uint8_t*>(av_mallocz(muxHeader_.size() + AV_INPUT_BUFFER_PADDING_SIZE));
    if (muxVideoStream_->codecpar->extradata == nullptr) {
      throw std::runtime_error("av_mallocz failed for MP4 codec extradata");
    }
    std::memcpy(muxVideoStream_->codecpar->extradata, muxHeader_.data(), muxHeader_.size());
    muxVideoStream_->codecpar->extradata_size = static_cast<int>(muxHeader_.size());
  }

  if ((muxFormatContext_->oformat->flags & AVFMT_NOFILE) == 0) {
    checkAvStatus(avio_open(&muxFormatContext_->pb, config.outputPath.c_str(), AVIO_FLAG_WRITE),
                  "avio_open failed");
  }
  AVDictionary* muxOptions = nullptr;
  if (outputRtsp) {
    av_dict_set(&muxOptions, "rtsp_transport", "tcp", 0);
    av_dict_set(&muxOptions, "muxdelay", "0.1", 0);
    av_dict_set(&muxOptions, "muxpreload", "0", 0);
  }
  packetTimebaseNum_ = muxVideoStream_->time_base.num;
  packetTimebaseDen_ = muxVideoStream_->time_base.den;
  packetDurationTicks_ = computePacketDurationTicks(fpsNum_, fpsDen_, muxVideoStream_->time_base);
  nextPacketPts_ = 0;
  const int headerStatus = avformat_write_header(muxFormatContext_, &muxOptions);
  av_dict_free(&muxOptions);
  checkAvStatus(headerStatus, "avformat_write_header failed");
  muxHeaderWritten_ = true;
}

void MppEncoder::closeOutputSink() {
  if (muxFormatContext_ != nullptr) {
    if (muxHeaderWritten_ && !rtspOutput_) {
      av_write_trailer(muxFormatContext_);
    }
    if ((muxFormatContext_->oformat->flags & AVFMT_NOFILE) == 0 && muxFormatContext_->pb != nullptr) {
      avio_closep(&muxFormatContext_->pb);
    }
    avformat_free_context(muxFormatContext_);
    muxFormatContext_ = nullptr;
    muxVideoStream_ = nullptr;
    muxHeader_.clear();
    muxHeaderWritten_ = false;
  }
  if (outputFile_.is_open()) {
    outputFile_.close();
  }
}

void MppEncoder::drainPackets(bool untilEos) {
  int retries = 0;
  while (true) {
    MppPacket packet = nullptr;
    checkMppStatus(api_->encode_get_packet(context_, &packet), "encode_get_packet failed");
    if (packet == nullptr) {
      const int maxRetries = untilEos ? kFlushDrainRetryCount : kDrainRetryCount;
      if (untilEos && retries++ < maxRetries) {
        std::this_thread::sleep_for(kDrainRetrySleep);
        continue;
      }
      return;
    }
    retries = 0;

    const bool packetEos = mpp_packet_get_eos(packet) != 0;
    writePacket(packet);
    mpp_packet_deinit(&packet);

    if (!untilEos || packetEos) {
      break;
    }
  }
}
