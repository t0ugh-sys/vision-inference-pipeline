#include "backends/mpp_encoder.hpp"

extern "C" {
#include <mpp_buffer.h>
#include <mpp_err.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
#include <rk_mpi.h>
#include <rk_mpi_cmd.h>
}

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {

constexpr size_t kPacketBufferMinSize = 512 * 1024;

void checkMppStatus(MPP_RET status, const char* message) {
  if (status != MPP_OK) {
    throw std::runtime_error(message);
  }
}

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

}  // namespace

MppEncoder::MppEncoder() = default;

MppEncoder::~MppEncoder() {
  close();
}

void MppEncoder::init(const EncoderConfig& config) {
  close();

  if (config.width <= 0 || config.height <= 0) {
    throw std::runtime_error("MPP encoder requires a positive frame size");
  }
  if (config.inputFormat != PixelFormat::kNv12) {
    throw std::runtime_error("MPP encoder currently only supports NV12 decoded-frame input");
  }

  width_ = config.width;
  height_ = config.height;
  horStride_ = config.horStride > 0 ? config.horStride : config.width;
  verStride_ = config.verStride > 0 ? config.verStride : config.height;
  inputFormat_ = config.inputFormat;
  flushSubmitted_ = false;

  outputFile_ = std::ofstream(config.outputPath, std::ios::binary);
  if (!outputFile_.is_open()) {
    throw std::runtime_error("Failed to open output file: " + config.outputPath);
  }

  checkMppStatus(mpp_buffer_group_get_internal(&bufferGroup_, MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE),
                 "mpp_buffer_group_get_internal failed");
  checkMppStatus(mpp_buffer_get(bufferGroup_, &packetBuffer_, packetBufferSize(config)),
                 "mpp_buffer_get for output packet failed");

  checkMppStatus(mpp_create(&context_, &api_), "mpp_create failed");
  checkMppStatus(mpp_init(context_, MPP_CTX_ENC, toCodingType(config.codec)), "mpp_init failed");

  MppPollType timeout = MPP_POLL_BLOCK;
  checkMppStatus(api_->control(context_, MPP_SET_OUTPUT_TIMEOUT, &timeout),
                 "MPP_SET_OUTPUT_TIMEOUT failed");

  checkMppStatus(mpp_enc_cfg_init(&config_), "mpp_enc_cfg_init failed");
  checkMppStatus(api_->control(context_, MPP_ENC_GET_CFG, config_), "MPP_ENC_GET_CFG failed");

  checkMppStatus(mpp_enc_cfg_set_s32(config_, "prep:width", width_), "prep:width failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "prep:height", height_), "prep:height failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "prep:hor_stride", horStride_), "prep:hor_stride failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "prep:ver_stride", verStride_), "prep:ver_stride failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "prep:format", MPP_FMT_YUV420SP), "prep:format failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "codec:type", toCodingType(config.codec)), "codec:type failed");

  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:mode", MPP_ENC_RC_MODE_CBR), "rc:mode failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:bps_target", config.bitrate), "rc:bps_target failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:bps_max", config.bitrate * 17 / 16), "rc:bps_max failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:bps_min", config.bitrate / 16), "rc:bps_min failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:fps_in_flex", 0), "rc:fps_in_flex failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:fps_in_num", config.fps), "rc:fps_in_num failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:fps_in_denom", 1), "rc:fps_in_denom failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:fps_out_flex", 0), "rc:fps_out_flex failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:fps_out_num", config.fps), "rc:fps_out_num failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:fps_out_denom", 1), "rc:fps_out_denom failed");
  checkMppStatus(mpp_enc_cfg_set_s32(config_, "rc:gop", std::max(1, config.fps * 2)), "rc:gop failed");

  if (config.codec == "hevc" || config.codec == "h265") {
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h265:qp_init", 26), "h265:qp_init failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h265:qp_max", 51), "h265:qp_max failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h265:qp_min", 10), "h265:qp_min failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h265:qp_max_i", 46), "h265:qp_max_i failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h265:qp_min_i", 18), "h265:qp_min_i failed");
  } else {
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:profile", 100), "h264:profile failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:level", 40), "h264:level failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:cabac_en", 1), "h264:cabac_en failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:cabac_idc", 0), "h264:cabac_idc failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:qp_init", 26), "h264:qp_init failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:qp_max", 51), "h264:qp_max failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:qp_min", 10), "h264:qp_min failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:qp_max_i", 46), "h264:qp_max_i failed");
    checkMppStatus(mpp_enc_cfg_set_s32(config_, "h264:qp_min_i", 18), "h264:qp_min_i failed");
  }

  checkMppStatus(api_->control(context_, MPP_ENC_SET_CFG, config_), "MPP_ENC_SET_CFG failed");

  MppPacket headerPacket = nullptr;
  checkMppStatus(mpp_packet_init_with_buffer(&headerPacket, packetBuffer_), "mpp_packet_init_with_buffer failed");
  mpp_packet_set_length(headerPacket, 0);
  checkMppStatus(api_->control(context_, MPP_ENC_GET_HDR_SYNC, headerPacket), "MPP_ENC_GET_HDR_SYNC failed");
  writePacket(headerPacket);
  mpp_packet_deinit(&headerPacket);

  initialized_ = true;
}

void MppEncoder::encode(const RgbImage& frame, int64_t pts) {
  (void)frame;
  (void)pts;
  throw std::runtime_error("MPP encoder expects DecodedFrame NV12 input, not CPU RGB frames");
}

void MppEncoder::encodeDecodedFrame(const DecodedFrame& frame, int64_t pts) {
  if (!initialized_) {
    throw std::runtime_error("MPP encoder is not initialized");
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
  MppPacket outputPacket = nullptr;

  try {
    MppBufferInfo bufferInfo;
    std::memset(&bufferInfo, 0, sizeof(bufferInfo));
    bufferInfo.type = MPP_BUFFER_TYPE_EXT_DMA;
    bufferInfo.fd = frame.dmaFd;
    bufferInfo.size = static_cast<size_t>(frame.horizontalStride) * static_cast<size_t>(frame.verticalStride) * 3 / 2;
    checkMppStatus(mpp_buffer_import(&inputBuffer, &bufferInfo), "mpp_buffer_import failed");

    checkMppStatus(mpp_frame_init(&inputFrame), "mpp_frame_init failed");
    mpp_frame_set_width(inputFrame, frame.width);
    mpp_frame_set_height(inputFrame, frame.height);
    mpp_frame_set_hor_stride(inputFrame, frame.horizontalStride > 0 ? frame.horizontalStride : horStride_);
    mpp_frame_set_ver_stride(inputFrame, frame.verticalStride > 0 ? frame.verticalStride : verStride_);
    mpp_frame_set_fmt(inputFrame, MPP_FMT_YUV420SP);
    mpp_frame_set_pts(inputFrame, pts);
    mpp_frame_set_buffer(inputFrame, inputBuffer);

    checkMppStatus(mpp_packet_init_with_buffer(&outputPacket, packetBuffer_), "mpp_packet_init_with_buffer failed");
    mpp_packet_set_length(outputPacket, 0);

    MppMeta meta = mpp_frame_get_meta(inputFrame);
    mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, outputPacket);

    checkMppStatus(api_->encode_put_frame(context_, inputFrame), "encode_put_frame failed");
    mpp_frame_deinit(&inputFrame);
    mpp_buffer_put(inputBuffer);
    inputBuffer = nullptr;

    drainPackets(false);
  } catch (...) {
    if (outputPacket != nullptr) {
      mpp_packet_deinit(&outputPacket);
    }
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

  MppFrame eosFrame = nullptr;
  try {
    checkMppStatus(mpp_frame_init(&eosFrame), "mpp_frame_init failed");
    mpp_frame_set_eos(eosFrame, 1);
    checkMppStatus(api_->encode_put_frame(context_, eosFrame), "encode_put_frame(eos) failed");
    flushSubmitted_ = true;
    mpp_frame_deinit(&eosFrame);
    drainPackets(true);
    outputFile_.flush();
  } catch (...) {
    if (eosFrame != nullptr) {
      mpp_frame_deinit(&eosFrame);
    }
    throw;
  }
}

void MppEncoder::close() {
  if (outputFile_.is_open()) {
    outputFile_.close();
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
  width_ = 0;
  height_ = 0;
  horStride_ = 0;
  verStride_ = 0;
  inputFormat_ = PixelFormat::kUnknown;
}

void MppEncoder::writePacket(void* opaquePacket) {
  MppPacket packet = static_cast<MppPacket>(opaquePacket);
  if (!packet) {
    return;
  }

  void* data = mpp_packet_get_pos(packet);
  const size_t length = mpp_packet_get_length(packet);
  if (data != nullptr && length > 0) {
    outputFile_.write(static_cast<const char*>(data), static_cast<std::streamsize>(length));
  }
}

void MppEncoder::drainPackets(bool untilEos) {
  while (true) {
    MppPacket packet = nullptr;
    checkMppStatus(api_->encode_get_packet(context_, &packet), "encode_get_packet failed");
    if (packet == nullptr) {
      if (untilEos) {
        continue;
      }
      return;
    }

    const bool packetEos = mpp_packet_get_eos(packet) != 0;
    writePacket(packet);
    mpp_packet_deinit(&packet);

    if (!untilEos || packetEos) {
      break;
    }
  }
}
