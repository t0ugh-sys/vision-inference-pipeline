#include "backends/mpp_decoder.hpp"

extern "C" {
#include <mpp_buffer.h>
#include <mpp_err.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
#include <rk_mpi.h>
}

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace {

constexpr MppCodingType kCodingAvc = MPP_VIDEO_CodingAVC;
constexpr MppCodingType kCodingHevc = MPP_VIDEO_CodingHEVC;
constexpr int kMaxPutAttempts = 32;
constexpr RK_S32 kFrameGroupCount = 12;

void checkMppStatus(MPP_RET status, const char* message) {
  if (status != MPP_OK) {
    throw std::runtime_error(message);
  }
}

struct MppFrameHolder {
  MppFrame frame = nullptr;

  ~MppFrameHolder() {
    if (frame != nullptr) {
      mpp_frame_deinit(&frame);
    }
  }
};

}  // namespace

MppDecoder::~MppDecoder() {
  close();
}

void MppDecoder::open(VideoCodec codec) {
  close();

  checkMppStatus(mpp_create(&context_, &api_), "mpp_create failed");
  checkMppStatus(
      mpp_init(context_, MPP_CTX_DEC, static_cast<MppCodingType>(toMppCodec(codec))),
      "mpp_init failed");

  RK_U32 splitMode = 1;
  checkMppStatus(
      api_->control(context_, MPP_DEC_SET_PARSER_SPLIT_MODE, &splitMode),
      "MPP_DEC_SET_PARSER_SPLIT_MODE failed");

  eosSubmitted_ = false;
}

int MppDecoder::toMppCodec(VideoCodec codec) const {
  switch (codec) {
    case VideoCodec::kH264:
      return kCodingAvc;
    case VideoCodec::kH265:
      return kCodingHevc;
    default:
      throw std::runtime_error("Unsupported codec for MPP");
  }
}

void MppDecoder::close() {
  readyFrames_.clear();
  eosSubmitted_ = false;
  if (externalBufferGroup_ != nullptr) {
    mpp_buffer_group_put(externalBufferGroup_);
    externalBufferGroup_ = nullptr;
  }
  if (context_ != nullptr) {
    mpp_destroy(context_);
  }
  context_ = nullptr;
  api_ = nullptr;
}

void MppDecoder::submitPacket(const EncodedPacket& packet) {
  MppPacket mppPacket = nullptr;
  checkMppStatus(
      mpp_packet_init(
          &mppPacket,
          packet.endOfStream ? nullptr : const_cast<std::uint8_t*>(packet.data.data()),
          packet.endOfStream ? 0 : packet.data.size()),
      "mpp_packet_init failed");

  if (!packet.endOfStream && !packet.data.empty()) {
    mpp_packet_set_pos(mppPacket, const_cast<std::uint8_t*>(packet.data.data()));
    mpp_packet_set_length(mppPacket, packet.data.size());
  }
  if (packet.endOfStream) {
    mpp_packet_set_eos(mppPacket);
    eosSubmitted_ = true;
  }
  mpp_packet_set_pts(mppPacket, packet.pts);

  bool submitted = false;
  for (int attempt = 0; attempt < kMaxPutAttempts; ++attempt) {
    const MPP_RET status = api_->decode_put_packet(context_, mppPacket);
    if (status == MPP_OK) {
      submitted = true;
      break;
    }
    if (status != MPP_ERR_BUFFER_FULL) {
      mpp_packet_deinit(&mppPacket);
      checkMppStatus(status, "decode_put_packet failed");
    }

    drainFramesToReadyQueue();
  }

  mpp_packet_deinit(&mppPacket);
  if (!submitted) {
    throw std::runtime_error("decode_put_packet stayed buffer-full and could not make progress");
  }
}

std::optional<DecodedFrame> MppDecoder::receiveFrame() {
  if (auto frame = popReadyFrame()) {
    return frame;
  }

  return decodeOneFrame();
}

std::optional<DecodedFrame> MppDecoder::popReadyFrame() {
  if (readyFrames_.empty()) {
    return std::nullopt;
  }

  DecodedFrame frame = std::move(readyFrames_.front());
  readyFrames_.pop_front();
  return frame;
}

std::optional<DecodedFrame> MppDecoder::decodeOneFrame() {
  while (true) {
    MppFrame frame = nullptr;
    const MPP_RET status = api_->decode_get_frame(context_, &frame);
    if (status != MPP_OK) {
      checkMppStatus(status, "decode_get_frame failed");
    }
    if (frame == nullptr) {
      return std::nullopt;
    }

    if (mpp_frame_get_info_change(frame) != 0) {
      handleInfoChange(frame);
      continue;
    }

    if (mpp_frame_get_errinfo(frame) != 0 || mpp_frame_get_discard(frame) != 0) {
      mpp_frame_deinit(&frame);
      continue;
    }

    MppBuffer buffer = mpp_frame_get_buffer(frame);
    if (buffer == nullptr) {
      mpp_frame_deinit(&frame);
      throw std::runtime_error("MPP frame buffer is null");
    }

    auto holder = std::make_shared<MppFrameHolder>();
    holder->frame = frame;

    DecodedFrame output;
    output.width = mpp_frame_get_width(frame);
    output.height = mpp_frame_get_height(frame);
    output.horizontalStride = mpp_frame_get_hor_stride(frame);
    output.verticalStride = mpp_frame_get_ver_stride(frame);
    output.chromaStride = output.horizontalStride;
    output.format = PixelFormat::kNv12;
    output.pts = mpp_frame_get_pts(frame);
    output.dmaFd = mpp_buffer_get_fd(buffer);
    output.nativeHandle = holder;
    return output;
  }
}

void MppDecoder::drainFramesToReadyQueue() {
  while (true) {
    auto frame = decodeOneFrame();
    if (!frame.has_value()) {
      break;
    }
    readyFrames_.push_back(std::move(*frame));
  }
}

void MppDecoder::handleInfoChange(void* opaqueFrame) {
  MppFrame frame = static_cast<MppFrame>(opaqueFrame);

  if (externalBufferGroup_ != nullptr) {
    mpp_buffer_group_put(externalBufferGroup_);
    externalBufferGroup_ = nullptr;
  }

  checkMppStatus(
      mpp_buffer_group_get_external(&externalBufferGroup_, MPP_BUFFER_TYPE_DRM),
      "mpp_buffer_group_get_external failed");

  size_t frameBytes = mpp_frame_get_buf_size(frame);
  if (frameBytes == 0) {
    const RK_U32 horStride = static_cast<RK_U32>(mpp_frame_get_hor_stride(frame));
    const RK_U32 verStride = static_cast<RK_U32>(mpp_frame_get_ver_stride(frame));
    const RK_U32 safeHorStride = horStride == 0 ? static_cast<RK_U32>(1) : horStride;
    const RK_U32 safeVerStride = verStride == 0 ? static_cast<RK_U32>(1) : verStride;
    frameBytes = static_cast<size_t>(safeHorStride) * static_cast<size_t>(safeVerStride) * 3 / 2;
  }

  checkMppStatus(
      mpp_buffer_group_limit_config(externalBufferGroup_, frameBytes, kFrameGroupCount),
      "mpp_buffer_group_limit_config failed");
  checkMppStatus(
      api_->control(context_, MPP_DEC_SET_EXT_BUF_GROUP, externalBufferGroup_),
      "MPP_DEC_SET_EXT_BUF_GROUP failed");
  checkMppStatus(
      api_->control(context_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr),
      "MPP_DEC_SET_INFO_CHANGE_READY failed");

  mpp_frame_deinit(&frame);
}
