#include "app_config.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<char*> makeArgv(std::vector<std::string>& arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) argv.push_back(argument.data());
  return argv;
}

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

bool testHelpRequest() {
  std::vector<std::string> arguments = {"video_pipeline", "--help"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kHelp, "expected help request to return kHelp") &&
         expect(result.message.find("--visual-style <classic|yolo>") != std::string::npos,
                "expected help output to include visual-style");
}

bool testBackendAndPositionals() {
  std::vector<std::string> arguments = {
      "video_pipeline", "--backend", "nvidia", "--gpu", "2", "--infer-workers", "3", "--max-frames", "99",
      "--conf-threshold", "0.3", "--nms-threshold", "0.5", "--postproc", "yolo26", "--labels-path", "labels.txt",
      "--letterbox", "false", "--verbose", "--dump-first-frame", "--model-output-layout",
      "yolov8_rknn_branch_6", "--display", "stream.mp4", "model.engine", "1280", "720"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kOk, "expected valid arguments to parse successfully") &&
         expect(result.config.decoderBackend == DecoderBackendType::kNvidiaNvdec, "expected nvidia decoder preset") &&
         expect(result.config.preprocBackend == PreprocBackendType::kNvidiaCuda, "expected nvidia preprocessor preset") &&
         expect(result.config.inferBackend == InferBackendType::kNvidiaTrt, "expected nvidia infer preset") &&
         expect(result.config.gpuId == 2, "expected gpu id to be parsed") &&
         expect(result.config.inferWorkers == 3, "expected infer workers to be parsed") &&
         expect(result.config.maxFrames == 99, "expected max frames to be parsed") &&
         expect(result.config.confThreshold == 0.3f, "expected conf threshold to be parsed") &&
         expect(result.config.nmsThreshold == 0.5f, "expected nms threshold to be parsed") &&
         expect(result.config.postprocBackend == PostprocBackendType::kYolo26, "expected postproc backend to be parsed") &&
         expect(result.config.labelsPath == "labels.txt", "expected labels path to be parsed") &&
         expect(!result.config.letterbox, "expected letterbox override to be parsed") &&
         expect(result.config.verbose, "expected verbose flag to be enabled") &&
         expect(result.config.dumpFirstFrame, "expected dump-first-frame flag to be enabled") &&
         expect(result.config.modelOutputLayout == ModelOutputLayout::kYolov8RknnBranch6, "expected output layout to be parsed") &&
         expect(result.config.visual.display, "expected display flag to be enabled") &&
         expect(result.config.source.uri == "stream.mp4", "expected source uri to be parsed") &&
         expect(result.config.model.modelPath == "model.engine", "expected model path to be parsed") &&
         expect(result.config.model.inputWidth == 1280, "expected width to be parsed") &&
         expect(result.config.model.inputHeight == 720, "expected height to be parsed");
}

bool testRejectOddPositionals() {
  std::vector<std::string> arguments = {"video_pipeline", "stream.mp4", "model.engine", "640"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kError, "expected missing height to fail") &&
         expect(result.message.find("Expected <video_or_rtsp> <model_file> [width] [height]") != std::string::npos, "expected positional usage hint");
}

bool testRejectUnknownOption() {
  std::vector<std::string> arguments = {"video_pipeline", "--unknown", "stream.mp4", "model.engine"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kError, "expected unknown option to fail") &&
         expect(result.message.find("Unknown option: --unknown") != std::string::npos, "expected unknown option message");
}

bool testRejectMissingOptionValue() {
  std::vector<std::string> arguments = {"video_pipeline", "--encoder-output"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kError, "expected missing option value to fail") &&
         expect(result.message.find("Missing value for --encoder-output") != std::string::npos,
                "expected missing option value message");
}

bool testRockchipStableOutputFlags() {
  std::vector<std::string> arguments = {
      "video_pipeline",
      "--backend", "rockchip",
      "--infer-workers", "2",
      "--rknn-zero-copy", "false",
      "--progress-every", "300",
      "--encoder-fps", "30",
      "--encoder-low-latency", "false",
      "--encoder-bitrate", "20000000",
      "--output-overlay", "rga",
      "--output-video", "/edge/workspace/vis_modelzoo_full.h264",
      "/edge/workspace/2_h264_clean.mp4",
      "/edge/workspace/rk-video-pipeline-cpp/models/stall_int8.rknn",
      "640",
      "640"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kOk, "expected stable rockchip flags to parse successfully") &&
         expect(result.config.decoderBackend == DecoderBackendType::kRockchipMpp, "expected rockchip decoder preset") &&
         expect(result.config.preprocBackend == PreprocBackendType::kRockchipRga, "expected rockchip preprocessor preset") &&
         expect(result.config.inferBackend == InferBackendType::kRockchipRknn, "expected rockchip infer preset") &&
         expect(result.config.inferWorkers == 2, "expected infer workers to be parsed") &&
         expect(!result.config.rknnZeroCopy, "expected rknn-zero-copy override to be parsed") &&
         expect(result.config.progressEvery == 300, "expected progress interval to be parsed") &&
         expect(result.config.encoderFps == 30, "expected encoder fps to be parsed") &&
         expect(result.config.encoderLowLatency == 0, "expected encoder low-latency override to be parsed") &&
         expect(result.config.encoderBitrate == 20000000, "expected encoder bitrate to be parsed") &&
         expect(result.config.visual.outputOverlayMode == OutputOverlayMode::kRga,
                "expected output overlay mode to be parsed") &&
         expect(result.config.visual.outputVideo == "/edge/workspace/vis_modelzoo_full.h264", "expected output video path to be parsed");
}

bool testInferWorkersAutoDefaultAndExplicitZero() {
  std::vector<std::string> defaultArguments = {
      "video_pipeline",
      "--backend", "rockchip",
      "stream.mp4",
      "model.rknn",
      "640",
      "640"};
  std::vector<char*> defaultArgv = makeArgv(defaultArguments);
  const ParseResult defaultResult =
      parseAppConfig(static_cast<int>(defaultArgv.size()), defaultArgv.data());

  std::vector<std::string> explicitArguments = {
      "video_pipeline",
      "--backend", "rockchip",
      "--infer-workers", "0",
      "stream.mp4",
      "model.rknn",
      "640",
      "640"};
  std::vector<char*> explicitArgv = makeArgv(explicitArguments);
  const ParseResult explicitResult =
      parseAppConfig(static_cast<int>(explicitArgv.size()), explicitArgv.data());

  return expect(defaultResult.status == ParseStatus::kOk, "expected default infer-workers to parse successfully") &&
         expect(defaultResult.config.inferWorkers == 0, "expected default infer-workers to remain auto") &&
         expect(explicitResult.status == ParseStatus::kOk, "expected explicit infer-workers 0 to parse successfully") &&
         expect(explicitResult.config.inferWorkers == 0, "expected explicit infer-workers 0 to mean auto");
}

bool testVisualStyleParsing() {
  std::vector<std::string> arguments = {
      "video_pipeline",
      "--backend", "rockchip",
      "--visual-style", "classic",
      "stream.mp4",
      "model.rknn",
      "640",
      "640"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kOk, "expected visual-style classic to parse successfully") &&
         expect(result.config.visual.style == VisualStyle::kClassic, "expected classic visual style to be parsed");
}

bool testRtspOutputParsing() {
  std::vector<std::string> arguments = {
      "video_pipeline",
      "--backend", "rockchip",
      "--output-rtsp", "rtsp://127.0.0.1:8554/live/test",
      "rtsp://127.0.0.1:554/input",
      "model.rknn",
      "640",
      "640"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kOk, "expected output-rtsp to parse successfully") &&
         expect(result.config.visual.outputRtsp == "rtsp://127.0.0.1:8554/live/test",
                "expected output rtsp url to be parsed") &&
         expect(result.config.source.uri == "rtsp://127.0.0.1:554/input",
                "expected input rtsp uri to be parsed");
}

bool testEncoderCodecAliasParsing() {
  std::vector<std::string> arguments = {
      "video_pipeline",
      "--backend", "rockchip",
      "--encoder-codec", "hevc",
      "stream.mp4",
      "model.rknn",
      "640",
      "640"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kOk, "expected encoder codec alias to parse successfully") &&
         expect(result.config.encoderCodec == "h265", "expected hevc alias to normalize to h265");
}

bool testRejectInvalidEncoderCodec() {
  std::vector<std::string> arguments = {
      "video_pipeline",
      "--backend", "rockchip",
      "--encoder-codec", "foo",
      "stream.mp4",
      "model.rknn",
      "640",
      "640"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kError, "expected invalid encoder codec to fail") &&
         expect(result.message.find("Unsupported encoder codec: foo") != std::string::npos,
                "expected invalid encoder codec message");
}

bool testRejectInvalidVisualStyle() {
  std::vector<std::string> arguments = {
      "video_pipeline",
      "--backend", "rockchip",
      "--visual-style", "invalid",
      "stream.mp4",
      "model.rknn",
      "640",
      "640"};
  std::vector<char*> argv = makeArgv(arguments);
  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kError, "expected invalid visual-style to fail") &&
         expect(result.message.find("Unsupported visual style: invalid") != std::string::npos,
                "expected invalid visual-style error message");
}

}  // namespace

int main() {
  bool ok = true;
  ok = ok && testHelpRequest();
  ok = ok && testBackendAndPositionals();
  ok = ok && testRejectOddPositionals();
  ok = ok && testRejectUnknownOption();
  ok = ok && testRejectMissingOptionValue();
  ok = ok && testRockchipStableOutputFlags();
  ok = ok && testInferWorkersAutoDefaultAndExplicitZero();
  ok = ok && testVisualStyleParsing();
  ok = ok && testRtspOutputParsing();
  ok = ok && testEncoderCodecAliasParsing();
  ok = ok && testRejectInvalidEncoderCodec();
  ok = ok && testRejectInvalidVisualStyle();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
