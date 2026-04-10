#include "backends/nvenc_encoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}

#include <cstring>
#include <stdexcept>

namespace {

void checkAvStatus(int status, const char* message) {
  if (status < 0) {
    char err[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(status, err, sizeof(err));
    throw std::runtime_error(std::string(message) + ": " + err);
  }
}

}  // namespace

NvencEncoder::NvencEncoder() = default;

NvencEncoder::~NvencEncoder() {
  close();
}

void NvencEncoder::init(const EncoderConfig& config) {
  close();

  // 打开输出文件
  outputFile_ = fopen(config.outputPath.c_str(), "wb");
  if (!outputFile_) {
    throw std::runtime_error("Failed to open output file: " + config.outputPath);
  }

  // 1. 创建 CUDA 硬件设备上下文
  int ret = av_hwdevice_ctx_create(
      &hwDeviceCtx_,
      AV_HWDEVICE_TYPE_CUDA,
      nullptr,
      nullptr,
      0);
  checkAvStatus(ret, "Failed to create CUDA hardware device context");

  // 2. 查找 NVENC 编码器
  const AVCodec* encoder = avcodec_find_encoder_by_name("h264_nvenc");
  if (!encoder) {
    throw std::runtime_error("NVENC encoder not found");
  }

  // 3. 创建编码器上下文
  codecCtx_ = avcodec_alloc_context3(encoder);
  if (!codecCtx_) {
    throw std::runtime_error("Failed to allocate codec context");
  }

  // 4. 设置编码参数
  codecCtx_->width = config.width;
  codecCtx_->height = config.height;
  codecCtx_->framerate = {config.fps, 1};
  codecCtx_->time_base = {1, config.fps};
  codecCtx_->bit_rate = config.bitrate;
  codecCtx_->gop_size = 30;
  codecCtx_->max_b_frames = 0;
  codecCtx_->pix_fmt = AV_PIX_FMT_CUDA;

  // NVENC 特定选项
  av_opt_set(codecCtx_->priv_data, "preset", "p4", 0);      // 低延迟 preset
  av_opt_set(codecCtx_->priv_data, "tune", "ull", 0);        // ultra-low-latency

  ret = avcodec_open2(codecCtx_, encoder, nullptr);
  checkAvStatus(ret, "Failed to open NVENC encoder");

  // 6. 创建帧缓冲区
  swFrame_ = av_frame_alloc();
  swFrame_->format = AV_PIX_FMT_BGR0;
  swFrame_->width = config.width;
  swFrame_->height = config.height;
  swFrame_->linesize[0] = config.width * 3;
  swFrame_->data[0] = static_cast<uint8_t*>(av_mallocz(swFrame_->linesize[0] * config.height));

  hwFrame_ = av_frame_alloc();
  hwFrame_->format = AV_PIX_FMT_CUDA;
  hwFrame_->width = config.width;
  hwFrame_->height = config.height;
  // 使用 hw_frames_ctx 替代 hw_device_ctx (FFmpeg 新版本)
  AVHWFramesContext* framesCtx = reinterpret_cast<AVHWFramesContext*>(
      av_mallocz(sizeof(AVHWFramesContext)));
  framesCtx->format = AV_PIX_FMT_CUDA;
  framesCtx->sw_format = AV_PIX_FMT_BGR0;
  framesCtx->width = config.width;
  framesCtx->height = config.height;
  framesCtx->device_ctx = hwDeviceCtx_;
  hwFrame_->hw_frames_ctx = av_buffer_create(
      reinterpret_cast<uint8_t*>(framesCtx), sizeof(AVHWFramesContext),
      nullptr, nullptr, 0);

  initialized_ = true;
}

void NvencEncoder::encode(const RgbImage& frame, int64_t pts) {
  if (!initialized_) {
    return;
  }

  // 复制数据到软件帧
  std::memcpy(swFrame_->data[0], frame.data.data(), frame.data.size());
  swFrame_->pts = pts;

  // 上传到 CUDA 硬件帧
  int ret = av_hwframe_transfer_data(hwFrame_, swFrame_, 0);
  if (ret < 0) {
    return;
  }

  // 编码
  ret = avcodec_send_frame(codecCtx_, hwFrame_);
  if (ret < 0) {
    return;
  }

  // 获取编码后的数据
  AVPacket* pkt = av_packet_alloc();
  while (ret >= 0) {
    ret = avcodec_receive_packet(codecCtx_, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    }
    if (ret < 0) {
      break;
    }

    // 写入文件
    fwrite(pkt->data, 1, pkt->size, outputFile_);
    av_packet_unref(pkt);
  }
  av_packet_free(&pkt);
}

void NvencEncoder::flush() {
  if (!initialized_) {
    return;
  }

  // 刷新编码器
  int ret = avcodec_send_frame(codecCtx_, nullptr);
  AVPacket* pkt = av_packet_alloc();
  while (ret >= 0) {
    ret = avcodec_receive_packet(codecCtx_, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    }
    if (ret < 0) {
      break;
    }

    fwrite(pkt->data, 1, pkt->size, outputFile_);
    av_packet_unref(pkt);
  }
  av_packet_free(&pkt);

  fflush(outputFile_);
}

void NvencEncoder::close() {
  if (outputFile_) {
    fclose(outputFile_);
    outputFile_ = nullptr;
  }
  if (swFrame_) {
    if (swFrame_->data[0]) {
      av_free(swFrame_->data[0]);
    }
    av_frame_free(&swFrame_);
  }
  if (hwFrame_) {
    av_frame_free(&hwFrame_);
  }
  if (codecCtx_) {
    avcodec_free_context(&codecCtx_);
  }
  if (hwDeviceCtx_) {
    av_buffer_unref(&hwDeviceCtx_);
  }
  initialized_ = false;
}
