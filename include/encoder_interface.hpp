#pragma once

#include "pipeline_types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/**
 * 编码器配置
 */
struct EncoderConfig {
  std::string outputPath;       // 输出文件路径 (mp4/h264)
  int width = 0;               // 视频宽度
  int height = 0;              // 视频高度
  int fps = 30;                // 帧率
  int bitrate = 2000000;       // 码率 (2Mbps)
  std::string codec = "h264";  // 编码器 (h264/hevc)
};

/**
 * 编码器后端接口
 */
class IEncoderBackend {
 public:
  virtual ~IEncoderBackend() = default;

  /**
   * 初始化编码器
   * @param config 编码配置
   */
  virtual void init(const EncoderConfig& config) = 0;

  /**
   * 编码一帧
   * @param frame RGB 图像
   * @param pts 时间戳
   */
  virtual void encode(const RgbImage& frame, int64_t pts) = 0;

  /**
   * 结束编码，写入文件
   */
  virtual void flush() = 0;

  /**
   * 释放资源
   */
  virtual void close() = 0;

  /** 获取后端名称 */
  virtual std::string name() const = 0;
};

/**
 * 编码器类型
 */
enum class EncoderBackendType {
  kAuto,           ///< 自动选择
  kNvidiaNvEnc,    ///< NVIDIA NVENC
  kRockchipMpp,    ///< Rockchip MPP 编码
  kCpu,            ///< CPU 软编码 (FFmpeg)
};

/**
 * 检测可用的编码器后端
 */
EncoderBackendType detectAvailableEncoderBackend();

/**
 * 创建编码器后端实例
 */
std::unique_ptr<IEncoderBackend> createEncoderBackend(EncoderBackendType type = EncoderBackendType::kAuto);
