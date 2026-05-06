#pragma once

#include "decoder_interface.hpp"
#include "infer_interface.hpp"
#include "postproc_interface.hpp"
#include "preproc_interface.hpp"
#include "visualizer.hpp"

#include <string>

struct AppConfig {
  InputSourceConfig source;
  ModelConfig model;
  DecoderBackendType decoderBackend = DecoderBackendType::kAuto;
  PreprocBackendType preprocBackend = PreprocBackendType::kAuto;
  InferBackendType inferBackend = InferBackendType::kAuto;
  PostprocBackendType postprocBackend = PostprocBackendType::kAuto;
  VisualConfig visual;
  float confThreshold = 0.25f;
  float nmsThreshold = 0.45f;
  std::string labelsPath;
  bool letterbox = true;
  bool verbose = false;
  bool dumpFirstFrame = false;
  ModelOutputLayout modelOutputLayout = ModelOutputLayout::kAuto;
  int inferWorkers = 0;
  int progressEvery = 30;
  bool rknnZeroCopy = true;
  RknnCoreMaskMode rknnCoreMask = RknnCoreMaskMode::kAuto;
  int gpuId = 0;
  int maxFrames = 0;
  bool outputOverlayExplicit = false;
  std::string encoderOutput;
  std::string encoderCodec = "h264";
  int encoderBitrate = 0;
  int encoderFps = 0;
  int encoderLowLatency = -1;
};

enum class ParseStatus {
  kOk,
  kHelp,
  kError,
};

struct ParseResult {
  ParseStatus status = ParseStatus::kError;
  AppConfig config;
  std::string message;
};

ParseResult parseAppConfig(int argc, char* argv[]);

std::string buildUsageMessage(const std::string& programName);
