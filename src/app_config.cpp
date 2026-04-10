#include "app_config.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int parseIntValue(const std::string& value, const char* optionName) {
  std::size_t parsedLength = 0;
  int parsedValue = 0;
  try {
    parsedValue = std::stoi(value, &parsedLength);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid integer for " + std::string(optionName) + ": " + value);
  }

  if (parsedLength != value.size()) {
    throw std::runtime_error("Invalid integer for " + std::string(optionName) + ": " + value);
  }

  return parsedValue;
}

void applyBackendPreset(const std::string& backendName, AppConfig& config) {
  if (backendName == "rockchip" || backendName == "mpp") {
    config.decoderBackend = DecoderBackendType::kRockchipMpp;
    config.preprocBackend = PreprocBackendType::kRockchipRga;
    config.inferBackend = InferBackendType::kRockchipRknn;
    return;
  }

  if (backendName == "nvidia" || backendName == "nvdec") {
    config.decoderBackend = DecoderBackendType::kNvidiaNvdec;
    config.preprocBackend = PreprocBackendType::kNvidiaCuda;
    config.inferBackend = InferBackendType::kNvidiaTrt;
    return;
  }

  throw std::runtime_error("Unsupported backend: " + backendName);
}

void assignPositionals(const std::vector<std::string>& positionals, AppConfig& config) {
  if (positionals.size() != 2 && positionals.size() != 4) {
    throw std::runtime_error(
        "Expected <video_or_rtsp> <model_file> [width] [height], but received " +
        std::to_string(positionals.size()) + " positional arguments");
  }

  config.source.uri = positionals[0];
  config.model.modelPath = positionals[1];

  if (positionals.size() == 4) {
    config.model.inputWidth = parseIntValue(positionals[2], "width");
    config.model.inputHeight = parseIntValue(positionals[3], "height");
    if (config.model.inputWidth <= 0 || config.model.inputHeight <= 0) {
      throw std::runtime_error("Model input width and height must be positive integers");
    }
  }
}

}  // namespace

ParseResult parseAppConfig(int argc, char* argv[]) {
  AppConfig config;

  if (argc <= 1) {
    return {
        ParseStatus::kError,
        config,
        "Missing required arguments.\n\n" + buildUsageMessage(argv[0]),
    };
  }

  std::vector<std::string> positionals;

  try {
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];

      if (argument == "--help" || argument == "-h") {
        return {ParseStatus::kHelp, config, buildUsageMessage(argv[0])};
      }

      if (argument == "--backend") {
        if (index + 1 >= argc) {
          throw std::runtime_error("Missing value for --backend");
        }
        applyBackendPreset(argv[++index], config);
        continue;
      }

      if (argument == "--gpu") {
        if (index + 1 >= argc) {
          throw std::runtime_error("Missing value for --gpu");
        }
        config.gpuId = parseIntValue(argv[++index], "--gpu");
        if (config.gpuId < 0) {
          throw std::runtime_error("--gpu must be greater than or equal to 0");
        }
        continue;
      }

      if (argument == "--max-frames") {
        if (index + 1 >= argc) {
          throw std::runtime_error("Missing value for --max-frames");
        }
        config.maxFrames = parseIntValue(argv[++index], "--max-frames");
        if (config.maxFrames < 0) {
          throw std::runtime_error("--max-frames must be greater than or equal to 0");
        }
        continue;
      }

      if (argument == "--display") {
        config.visual.display = true;
        continue;
      }

      if (argument == "--output-video") {
        if (index + 1 >= argc) {
          throw std::runtime_error("Missing value for --output-video");
        }
        config.visual.outputVideo = argv[++index];
        continue;
      }

      if (argument == "--output-rtsp") {
        if (index + 1 >= argc) {
          throw std::runtime_error("Missing value for --output-rtsp");
        }
        config.visual.outputRtsp = argv[++index];
        continue;
      }

      if (!argument.empty() && argument[0] == '-') {
        throw std::runtime_error("Unknown option: " + argument);
      }

      positionals.push_back(argument);
    }

    assignPositionals(positionals, config);
  } catch (const std::exception& error) {
    return {
        ParseStatus::kError,
        AppConfig{},
        std::string("Error: ") + error.what() + "\n\n" + buildUsageMessage(argv[0]),
    };
  }

  return {ParseStatus::kOk, config, {}};
}

std::string buildUsageMessage(const std::string& programName) {
  std::string message;
  message += "Usage: " + programName + " [options] <video_or_rtsp> <model_file> [width] [height]\n\n";
  message += "Options:\n";
  message += "  --backend <rockchip|mpp|nvidia|nvdec>  Select backend preset\n";
  message += "  --gpu <id>                              GPU device id (default: 0)\n";
  message += "  --max-frames <n>                        Max frames to process, 0 means unlimited\n";
  message += "  --display                               Enable display window\n";
  message += "  --output-video <path>                   Write annotated video to a file\n";
  message += "  --output-rtsp <url>                     Stream annotated video to RTSP\n";
  message += "  -h, --help                              Show this help message\n\n";
  message += "Examples:\n";
  message += "  " + programName + " --backend rockchip test.mp4 yolov5s.rknn 640 640\n";
  message += "  " + programName + " --backend nvidia --display test.mp4 yolov5s.engine 640 640\n";
  message += "  " + programName + " --backend nvidia --output-video output.mp4 test.mp4 yolov5s.engine 640 640\n";
  return message;
}
