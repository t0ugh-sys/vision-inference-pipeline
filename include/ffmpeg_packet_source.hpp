#pragma once

#include "pipeline_types.hpp"

#include <string>

struct AVFormatContext;

class FFmpegPacketSource {
 public:
  FFmpegPacketSource() = default;
  ~FFmpegPacketSource();

  FFmpegPacketSource(const FFmpegPacketSource&) = delete;
  FFmpegPacketSource& operator=(const FFmpegPacketSource&) = delete;

  void open(const InputSourceConfig& config);
  EncodedPacket readPacket();
  VideoCodec codec() const;

 private:
  static VideoCodec toVideoCodec(int codecId);
  void close();

  AVFormatContext* formatContext_ = nullptr;
  int videoStreamIndex_ = -1;
  VideoCodec codec_ = VideoCodec::kUnknown;
};
