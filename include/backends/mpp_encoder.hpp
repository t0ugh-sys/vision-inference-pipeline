#pragma once

#include "../encoder_interface.hpp"

extern "C" {
#include <mpp_frame.h>
#include <rk_type.h>
#include <rk_venc_cfg.h>
}

#include <fstream>
#include <cstddef>
#include <cstdint>
#include <string>
#include <mutex>
#include <vector>

struct MppApi_t;
typedef struct MppApi_t MppApi;
struct AVFormatContext;
struct AVStream;

class MppEncoder : public IEncoderBackend {
 public:
  MppEncoder();
  ~MppEncoder() override;

  void init(const EncoderConfig& config) override;
  void encode(const RgbImage& frame, int64_t pts) override;
  bool supportsDecodedFrameInput() const override { return true; }
  void encodeDecodedFrame(const DecodedFrame& frame, int64_t pts) override;
  void flush() override;
  void close() override;

  std::string name() const override { return "MPP Encoder"; }

 private:
  void initOutputSink(const EncoderConfig& config);
  void initMp4Muxer(const EncoderConfig& config);
  void writePacket(void* packet);
  void drainPackets(bool untilEos);
  void closeOutputSink();
  bool writesMp4Container() const { return muxFormatContext_ != nullptr; }

 MppCtx context_ = nullptr;
  MppApi* api_ = nullptr;
  MppEncCfg config_ = nullptr;
  MppBufferGroup bufferGroup_ = nullptr;
  MppBuffer inputBuffer_ = nullptr;
  MppBuffer packetBuffer_ = nullptr;
  std::ofstream outputFile_;
  AVFormatContext* muxFormatContext_ = nullptr;
  AVStream* muxVideoStream_ = nullptr;
  std::vector<std::uint8_t> muxHeader_;
  std::vector<std::uint8_t> muxPacketScratch_;
  std::mutex rgaMutex_;
  bool initialized_ = false;
  bool flushSubmitted_ = false;
  bool outputMuxingRequested_ = false;
  bool muxHeaderWritten_ = false;
  bool rtspOutput_ = false;
  bool outputDisconnected_ = false;
  bool timingEnabled_ = false;
  int width_ = 0;
  int height_ = 0;
  int horStride_ = 0;
  int verStride_ = 0;
  int64_t frameIndex_ = 0;
  MppFrameFormat frameFormat_ = MPP_FMT_YUV420SP;
  int rgaYuvFormat_ = 0;
  PixelFormat inputFormat_ = PixelFormat::kUnknown;
  int fpsNum_ = 30;
  int fpsDen_ = 1;
  int64_t nextPacketPts_ = 0;
  int64_t packetDurationTicks_ = 1;
  int packetTimebaseNum_ = 1;
  int packetTimebaseDen_ = 1;
  std::size_t timedFrameCount_ = 0;
  double totalEncodePutMs_ = 0.0;
  double totalEncodeGetMs_ = 0.0;
  double totalWritePacketMs_ = 0.0;
};
