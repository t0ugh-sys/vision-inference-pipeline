#pragma once

#include "decoder_interface.hpp"
#include "pipeline_types.hpp"

extern "C" {
#include <rk_type.h>
}

#include <deque>
#include <optional>

struct MppApi_t;
typedef struct MppApi_t MppApi;

class MppDecoder : public IDecoderBackend {
 public:
  MppDecoder() = default;
  ~MppDecoder() override;

  MppDecoder(const MppDecoder&) = delete;
  MppDecoder& operator=(const MppDecoder&) = delete;

  void open(VideoCodec codec) override;
  void submitPacket(const EncodedPacket& packet) override;
  std::optional<DecodedFrame> receiveFrame() override;
  std::string name() const override { return "Rockchip MPP"; }

 private:
  int toMppCodec(VideoCodec codec) const;
  void close();
  std::optional<DecodedFrame> popReadyFrame();
  std::optional<DecodedFrame> decodeOneFrame();
  void drainFramesToReadyQueue();
  void handleInfoChange(void* frame);

  MppCtx context_ = nullptr;
  MppApi* api_ = nullptr;
  MppBufferGroup externalBufferGroup_ = nullptr;
  std::deque<DecodedFrame> readyFrames_;
  bool eosSubmitted_ = false;
};
