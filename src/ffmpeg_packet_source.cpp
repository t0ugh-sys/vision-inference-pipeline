#include "ffmpeg_packet_source.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <stdexcept>

namespace {

[[noreturn]] void throwFfmpegError(const std::string& message, int errorCode) {
  char errorBuffer[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(errorCode, errorBuffer, sizeof(errorBuffer));
  throw std::runtime_error(message + ": " + errorBuffer);
}

}  // namespace

FFmpegPacketSource::~FFmpegPacketSource() {
  close();
}

void FFmpegPacketSource::open(const InputSourceConfig& config) {
  close();
  avformat_network_init();

  int result = avformat_open_input(&formatContext_, config.uri.c_str(), nullptr, nullptr);
  if (result < 0) {
    throwFfmpegError("Failed to open input", result);
  }

  result = avformat_find_stream_info(formatContext_, nullptr);
  if (result < 0) {
    throwFfmpegError("Failed to find stream info", result);
  }

  videoStreamIndex_ = av_find_best_stream(formatContext_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (videoStreamIndex_ < 0) {
    throw std::runtime_error("Failed to find video stream");
  }

  const AVStream* videoStream = formatContext_->streams[videoStreamIndex_];
  codec_ = toVideoCodec(videoStream->codecpar->codec_id);
  if (codec_ == VideoCodec::kUnknown) {
    throw std::runtime_error("Unsupported video codec for MPP decoder");
  }
}

EncodedPacket FFmpegPacketSource::readPacket() {
  AVPacket packet{};

  while (true) {
    const int result = av_read_frame(formatContext_, &packet);
    if (result == AVERROR_EOF) {
      EncodedPacket eosPacket;
      eosPacket.endOfStream = true;
      return eosPacket;
    }
    if (result < 0) {
      throwFfmpegError("Failed to read frame", result);
    }

    if (packet.stream_index != videoStreamIndex_) {
      av_packet_unref(&packet);
      continue;
    }

    // FFmpeg 的 packet 生命周期只到当前作用域，拷贝一份后再交给后续解码模块。
    EncodedPacket output;
    output.data.assign(packet.data, packet.data + packet.size);
    output.pts = packet.pts;
    output.keyFrame = (packet.flags & AV_PKT_FLAG_KEY) != 0;
    av_packet_unref(&packet);
    return output;
  }
}

VideoCodec FFmpegPacketSource::codec() const {
  return codec_;
}

VideoCodec FFmpegPacketSource::toVideoCodec(int codecId) {
  switch (codecId) {
    case AV_CODEC_ID_H264:
      return VideoCodec::kH264;
    case AV_CODEC_ID_HEVC:
      return VideoCodec::kH265;
    default:
      return VideoCodec::kUnknown;
  }
}

void FFmpegPacketSource::close() {
  if (formatContext_ != nullptr) {
    avformat_close_input(&formatContext_);
  }
  videoStreamIndex_ = -1;
  codec_ = VideoCodec::kUnknown;
}
