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

RknnCoreMaskMode parseRknnCoreMask(const std::string& value) {
  if (value == "auto") return RknnCoreMaskMode::kAuto;
  if (value == "0") return RknnCoreMaskMode::kCore0;
  if (value == "1") return RknnCoreMaskMode::kCore1;
  if (value == "2") return RknnCoreMaskMode::kCore2;
  if (value == "0_1") return RknnCoreMaskMode::kCore0_1;
  if (value == "0_2") return RknnCoreMaskMode::kCore0_2;
  if (value == "1_2") return RknnCoreMaskMode::kCore1_2;
  if (value == "0_1_2") return RknnCoreMaskMode::kCore0_1_2;
  if (value == "all") return RknnCoreMaskMode::kAll;
  throw std::runtime_error("Unsupported RKNN core mask: " + value);
}

ModelOutputLayout parseModelOutputLayout(const std::string& value) {
  if (value == "auto") return ModelOutputLayout::kAuto;
  if (value == "yolov8_flat_8400x84") return ModelOutputLayout::kYolov8Flat;
  if (value == "yolov8_rknn_branch_6") return ModelOutputLayout::kYolov8RknnBranch6;
  if (value == "yolov8_rknn_branch_9") return ModelOutputLayout::kYolov8RknnBranch9;
  if (value == "yolo26_e2e") return ModelOutputLayout::kYolo26E2E;
  throw std::runtime_error("Unsupported model output layout: " + value);
}

OutputOverlayMode parseOutputOverlayMode(const std::string& value) {
  if (value == "cpu") return OutputOverlayMode::kCpu;
  if (value == "rga") return OutputOverlayMode::kRga;
  throw std::runtime_error("Unsupported output overlay mode: " + value);
}

VisualStyle parseVisualStyle(const std::string& value) {
  if (value == "classic") return VisualStyle::kClassic;
  if (value == "yolo") return VisualStyle::kYolo;
  throw std::runtime_error("Unsupported visual style: " + value);
}

PostprocBackendType parsePostprocBackend(const std::string& value) {
  if (value == "auto") return PostprocBackendType::kAuto;
  if (value == "yolov8") return PostprocBackendType::kYoloV8;
  if (value == "yolo26") return PostprocBackendType::kYolo26;
  if (value == "yolov5") return PostprocBackendType::kYoloV5;
  throw std::runtime_error("Unsupported postprocessor backend: " + value);
}

std::string parseEncoderCodec(const std::string& value) {
  if (value == "h264" || value == "avc") return "h264";
  if (value == "h265" || value == "hevc") return "h265";
  throw std::runtime_error("Unsupported encoder codec: " + value);
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
      if (argument == "--progress-every") { config.progressEvery = parseIntValue(requireNextArg(argc, argv, index, "--progress-every"), "--progress-every"); continue; }
      if (argument == "--rknn-core-mask") { config.rknnCoreMask = parseRknnCoreMask(requireNextArg(argc, argv, index, "--rknn-core-mask")); continue; }
      if (argument == "--max-frames") { config.maxFrames = parseIntValue(requireNextArg(argc, argv, index, "--max-frames"), "--max-frames"); continue; }
      if (argument == "--conf-threshold") { config.confThreshold = parseFloatValue(requireNextArg(argc, argv, index, "--conf-threshold"), "--conf-threshold"); continue; }
      if (argument == "--nms-threshold") { config.nmsThreshold = parseFloatValue(requireNextArg(argc, argv, index, "--nms-threshold"), "--nms-threshold"); continue; }
      if (argument == "--postproc") { config.postprocBackend = parsePostprocBackend(requireNextArg(argc, argv, index, "--postproc")); continue; }
      if (argument == "--labels-path") { config.labelsPath = requireNextArg(argc, argv, index, "--labels-path"); continue; }
      if (argument == "--letterbox") { config.letterbox = parseBoolValue(requireNextArg(argc, argv, index, "--letterbox"), "--letterbox"); continue; }
      if (argument == "--rknn-zero-copy") { config.rknnZeroCopy = parseBoolValue(requireNextArg(argc, argv, index, "--rknn-zero-copy"), "--rknn-zero-copy"); continue; }
      if (argument == "--verbose") { config.verbose = true; continue; }
      if (argument == "--dump-first-frame") { config.dumpFirstFrame = true; continue; }
      if (argument == "--model-output-layout") { config.modelOutputLayout = parseModelOutputLayout(requireNextArg(argc, argv, index, "--model-output-layout")); continue; }
      if (argument == "--display") { config.visual.display = true; continue; }
      if (argument == "--display-max-width") { config.visual.displayMaxWidth = parseIntValue(requireNextArg(argc, argv, index, "--display-max-width"), "--display-max-width"); continue; }
      if (argument == "--display-max-height") { config.visual.displayMaxHeight = parseIntValue(requireNextArg(argc, argv, index, "--display-max-height"), "--display-max-height"); continue; }
      if (argument == "--output-overlay") {
        config.visual.outputOverlayMode = parseOutputOverlayMode(requireNextArg(argc, argv, index, "--output-overlay"));
        config.outputOverlayExplicit = true;
        continue;
      }
      if (argument == "--visual-style") { config.visual.style = parseVisualStyle(requireNextArg(argc, argv, index, "--visual-style")); continue; }
      if (argument == "--output-video") { config.visual.outputVideo = requireNextArg(argc, argv, index, "--output-video"); continue; }
      if (argument == "--output-rtsp") { config.visual.outputRtsp = requireNextArg(argc, argv, index, "--output-rtsp"); continue; }
      if (argument == "--encoder-output") { config.encoderOutput = requireNextArg(argc, argv, index, "--encoder-output"); continue; }
      if (argument == "--encoder-codec") { config.encoderCodec = parseEncoderCodec(requireNextArg(argc, argv, index, "--encoder-codec")); continue; }
      if (argument == "--encoder-bitrate") { config.encoderBitrate = parseIntValue(requireNextArg(argc, argv, index, "--encoder-bitrate"), "--encoder-bitrate"); continue; }
      if (argument == "--encoder-fps") { config.encoderFps = parseIntValue(requireNextArg(argc, argv, index, "--encoder-fps"), "--encoder-fps"); continue; }
      if (argument == "--encoder-low-latency") { config.encoderLowLatency = parseBoolValue(requireNextArg(argc, argv, index, "--encoder-low-latency"), "--encoder-low-latency") ? 1 : 0; continue; }
      if (!argument.empty() && argument[0] == '-') throw std::runtime_error("Unknown option: " + argument);
      positionals.push_back(argument);
    }
    assignPositionals(positionals, config);
    if (config.inferWorkers < 0) {
      throw std::runtime_error("--infer-workers must be greater than or equal to 0");
    }
    if (config.progressEvery <= 0) {
      throw std::runtime_error("--progress-every must be greater than 0");
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
  message += "  --infer-workers <n>                     Number of parallel inference workers (default: 0 = auto; Rockchip=3, others=1)\n";
  message += "  --progress-every <n>                    Print one progress log every n frames (default: 30)\n";
  message += "  --rknn-core-mask <mask>                 auto|0|1|2|0_1|0_2|1_2|0_1_2|all\n";
  message += "  --max-frames <n>                        Max frames to process (default: 0 = unlimited)\n";
  message += "  --conf-threshold <f>                    Detection confidence threshold\n";
  message += "  --nms-threshold <f>                     NMS IoU threshold\n";
  message += "  --postproc <auto|yolov8|yolo26|yolov5> Postprocessor backend\n";
  message += "  --labels-path <path>                    Optional labels file path\n";
  message += "  --letterbox <true|false>                Enable letterbox preprocessing\n";
  message += "  --rknn-zero-copy <true|false>           Prefer DMA RGB input for RKNN, fallback to host-copy on failure\n";
  message += "  --model-output-layout <name>            auto|yolov8_flat_8400x84|yolov8_rknn_branch_6|yolov8_rknn_branch_9|yolo26_e2e (unsupported)\n";
  message += "  --verbose                               Enable verbose logs\n";
  message += "  --dump-first-frame                      Dump first inference input frame\n";
  message += "  --display                               Enable display window\n";
  message += "  --display-max-width <n>                 Max display-path width, 0 keeps source width\n";
  message += "  --display-max-height <n>                Max display-path height, 0 keeps source height\n";
  message += "  --output-overlay <cpu|rga>              Overlay mode for --output-video/--output-rtsp (Rockchip annotated output auto: rga)\n";
  message += "  --visual-style <classic|yolo>           Detection label style (default: yolo)\n";
  message += "  --output-video <path>                   Write annotated video to a file (.h264/.264/.mp4 on Rockchip)\n";
  message += "  --output-rtsp <url>                     Stream annotated video to an RTSP server\n";
  message += "  --encoder-output <path>                 Write encoded output stream (Rockchip currently h264 only)\n";
  message += "  --encoder-codec <h264|h265>             Encoder codec (default: h264; Rockchip output currently h264 only)\n";
  message += "  --encoder-bitrate <bps>                 Encoder bitrate (default: 0 = auto)\n";
  message += "  --encoder-fps <n>                       Encoder fps (default: 0 = source fps)\n";
  message += "  --encoder-low-latency <true|false>      Low-latency encoder profile (default: auto; RTSP=true)\n";
  message += "  -h, --help                              Show this help message\n";
  return message;
}
