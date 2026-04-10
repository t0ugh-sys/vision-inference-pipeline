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
  int gpuId = 0;
  int maxFrames = 30;
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
