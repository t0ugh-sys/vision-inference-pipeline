#include "app_config.hpp"

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

float parseFloatValue(const std::string& value, const char* optionName) {
  std::size_t parsedLength = 0;
  float parsedValue = 0.0f;
  try {
    parsedValue = std::stof(value, &parsedLength);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid float for " + std::string(optionName) + ": " + value);
  }
  if (parsedLength != value.size()) {
    throw std::runtime_error("Invalid float for " + std::string(optionName) + ": " + value);
  }
  return parsedValue;
}

bool parseBoolValue(const std::string& value, const char* optionName) {
  if (value == "1" || value == "true" || value == "on") return true;
  if (value == "0" || value == "false" || value == "off") return false;
  throw std::runtime_error("Invalid boolean for " + std::string(optionName) + ": " + value);
}

ModelOutputLayout parseModelOutputLayout(const std::string& value) {
  if (value == "auto") return ModelOutputLayout::kAuto;
  if (value == "yolov8_flat_8400x84") return ModelOutputLayout::kYolov8Flat;
  if (value == "yolov8_rknn_branch_6") return ModelOutputLayout::kYolov8RknnBranch6;
  if (value == "yolov8_rknn_branch_9") return ModelOutputLayout::kYolov8RknnBranch9;
  if (value == "yolo26_e2e") return ModelOutputLayout::kYolo26E2E;
  throw std::runtime_error("Unsupported model output layout: " + value);
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
    throw std::runtime_error("Expected <video_or_rtsp> <model_file> [width] [height], but received " + std::to_string(positionals.size()) + " positional arguments");
  }
  config.source.uri = positionals[0];
  config.model.modelPath = positionals[1];
  if (positionals.size() == 4) {
    config.model.inputWidth = parseIntValue(positionals[2], "width");
    config.model.inputHeight = parseIntValue(positionals[3], "height");
  }
}

const char* requireNextArg(int argc, char* argv[], int& index, const char* optionName) {
  if (index + 1 >= argc) {
    throw std::runtime_error("Missing value for " + std::string(optionName));
  }
  ++index;
  return argv[index];
}

}  // namespace

ParseResult parseAppConfig(int argc, char* argv[]) {
  AppConfig config;
  if (argc <= 1) {
    return {ParseStatus::kError, config, "Missing required arguments.\n\n" + buildUsageMessage(argv[0])};
  }

  std::vector<std::string> positionals;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--help" || argument == "-h") return {ParseStatus::kHelp, config, buildUsageMessage(argv[0])};
      if (argument == "--backend") { applyBackendPreset(requireNextArg(argc, argv, index, "--backend"), config); continue; }
      if (argument == "--gpu") { config.gpuId = parseIntValue(requireNextArg(argc, argv, index, "--gpu"), "--gpu"); continue; }
      if (argument == "--infer-workers") { config.inferWorkers = parseIntValue(requireNextArg(argc, argv, index, "--infer-workers"), "--infer-workers"); continue; }
      if (argument == "--max-frames") { config.maxFrames = parseIntValue(requireNextArg(argc, argv, index, "--max-frames"), "--max-frames"); continue; }
      if (argument == "--conf-threshold") { config.confThreshold = parseFloatValue(requireNextArg(argc, argv, index, "--conf-threshold"), "--conf-threshold"); continue; }
      if (argument == "--nms-threshold") { config.nmsThreshold = parseFloatValue(requireNextArg(argc, argv, index, "--nms-threshold"), "--nms-threshold"); continue; }
      if (argument == "--labels-path") { config.labelsPath = requireNextArg(argc, argv, index, "--labels-path"); continue; }
      if (argument == "--letterbox") { config.letterbox = parseBoolValue(requireNextArg(argc, argv, index, "--letterbox"), "--letterbox"); continue; }
      if (argument == "--verbose") { config.verbose = true; continue; }
      if (argument == "--dump-first-frame") { config.dumpFirstFrame = true; continue; }
      if (argument == "--model-output-layout") { config.modelOutputLayout = parseModelOutputLayout(requireNextArg(argc, argv, index, "--model-output-layout")); continue; }
      if (argument == "--display") { config.visual.display = true; continue; }
      if (argument == "--output-video") { config.visual.outputVideo = requireNextArg(argc, argv, index, "--output-video"); continue; }
      if (argument == "--output-rtsp") { config.visual.outputRtsp = requireNextArg(argc, argv, index, "--output-rtsp"); continue; }
      if (argument == "--encoder-output") { config.encoderOutput = requireNextArg(argc, argv, index, "--encoder-output"); continue; }
      if (argument == "--encoder-codec") { config.encoderCodec = requireNextArg(argc, argv, index, "--encoder-codec"); continue; }
      if (argument == "--encoder-bitrate") { config.encoderBitrate = parseIntValue(requireNextArg(argc, argv, index, "--encoder-bitrate"), "--encoder-bitrate"); continue; }
      if (argument == "--encoder-fps") { config.encoderFps = parseIntValue(requireNextArg(argc, argv, index, "--encoder-fps"), "--encoder-fps"); continue; }
      if (!argument.empty() && argument[0] == '-') throw std::runtime_error("Unknown option: " + argument);
      positionals.push_back(argument);
    }
    assignPositionals(positionals, config);
    if (config.inferWorkers <= 0) {
      throw std::runtime_error("--infer-workers must be greater than 0");
    }
  } catch (const std::exception& error) {
    return {ParseStatus::kError, AppConfig{}, std::string("Error: ") + error.what() + "\n\n" + buildUsageMessage(argv[0])};
  }
  return {ParseStatus::kOk, config, {}};
}

std::string buildUsageMessage(const std::string& programName) {
  std::string message;
  message += "Usage: " + programName + " [options] <video_or_rtsp> <model_file> [width] [height]\n\n";
  message += "Options:\n";
  message += "  --backend <rockchip|mpp|nvidia|nvdec>  Select backend preset\n";
  message += "  --gpu <id>                              GPU device id\n";
  message += "  --infer-workers <n>                     Number of parallel inference workers\n";
  message += "  --max-frames <n>                        Max frames to process\n";
  message += "  --conf-threshold <f>                    Detection confidence threshold\n";
  message += "  --nms-threshold <f>                     NMS IoU threshold\n";
  message += "  --labels-path <path>                    Optional labels file path\n";
  message += "  --letterbox <true|false>                Enable letterbox preprocessing\n";
  message += "  --model-output-layout <name>            auto|yolov8_flat_8400x84|yolov8_rknn_branch_6|yolov8_rknn_branch_9|yolo26_e2e\n";
  message += "  --verbose                               Enable verbose logs\n";
  message += "  --dump-first-frame                      Dump first inference input frame\n";
  message += "  --display                               Enable display window\n";
  message += "  --output-video <path>                   Write annotated video to a file\n";
  message += "  --output-rtsp <url>                     Stream annotated video to RTSP\n";
  message += "  --encoder-output <path>                 Write raw decode stream (MPP h264/h265)\n";
  message += "  --encoder-codec <h264|h265>             Encoder codec (default: h264)\n";
  message += "  --encoder-bitrate <bps>                 Encoder bitrate (default: 4000000)\n";
  message += "  --encoder-fps <n>                       Encoder fps (default: 30)\n";
  message += "  -h, --help                              Show this help message\n";
  return message;
}
