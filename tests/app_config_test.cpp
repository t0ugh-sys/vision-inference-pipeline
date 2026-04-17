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
  return expect(result.status == ParseStatus::kHelp, "expected help request to return kHelp");
}

bool testBackendAndPositionals() {
  std::vector<std::string> arguments = {
      "video_pipeline", "--backend", "nvidia", "--gpu", "2", "--infer-workers", "3", "--max-frames", "99",
      "--conf-threshold", "0.3", "--nms-threshold", "0.5", "--labels-path", "labels.txt",
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

}  // namespace

int main() {
  bool ok = true;
  ok = ok && testHelpRequest();
  ok = ok && testBackendAndPositionals();
  ok = ok && testRejectOddPositionals();
  ok = ok && testRejectUnknownOption();
  ok = ok && testRejectMissingOptionValue();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
