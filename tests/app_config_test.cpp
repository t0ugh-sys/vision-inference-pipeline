#include "app_config.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<char*> makeArgv(std::vector<std::string>& arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) {
    argv.push_back(argument.data());
  }
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
      "video_pipeline",
      "--backend",
      "nvidia",
      "--gpu",
      "2",
      "--max-frames",
      "99",
      "--display",
      "stream.mp4",
      "model.engine",
      "1280",
      "720",
  };
  std::vector<char*> argv = makeArgv(arguments);

  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kOk, "expected valid arguments to parse successfully") &&
         expect(result.config.decoderBackend == DecoderBackendType::kNvidiaNvdec, "expected nvidia decoder preset") &&
         expect(result.config.preprocBackend == PreprocBackendType::kNvidiaCuda, "expected nvidia preprocessor preset") &&
         expect(result.config.inferBackend == InferBackendType::kNvidiaTrt, "expected nvidia infer preset") &&
         expect(result.config.gpuId == 2, "expected gpu id to be parsed") &&
         expect(result.config.maxFrames == 99, "expected max frames to be parsed") &&
         expect(result.config.visual.display, "expected display flag to be enabled") &&
         expect(result.config.source.uri == "stream.mp4", "expected source uri to be parsed") &&
         expect(result.config.model.modelPath == "model.engine", "expected model path to be parsed") &&
         expect(result.config.model.inputWidth == 1280, "expected width to be parsed") &&
         expect(result.config.model.inputHeight == 720, "expected height to be parsed");
}

bool testRejectOddPositionals() {
  std::vector<std::string> arguments = {
      "video_pipeline",
      "stream.mp4",
      "model.engine",
      "640",
  };
  std::vector<char*> argv = makeArgv(arguments);

  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kError, "expected missing height to fail") &&
         expect(result.message.find("Expected <video_or_rtsp> <model_file> [width] [height]") != std::string::npos,
                "expected usage hint for positional arguments");
}

bool testRejectUnknownOption() {
  std::vector<std::string> arguments = {
      "video_pipeline",
      "--unknown",
      "stream.mp4",
      "model.engine",
  };
  std::vector<char*> argv = makeArgv(arguments);

  const ParseResult result = parseAppConfig(static_cast<int>(argv.size()), argv.data());
  return expect(result.status == ParseStatus::kError, "expected unknown option to fail") &&
         expect(result.message.find("Unknown option: --unknown") != std::string::npos,
                "expected unknown option message");
}

}  // namespace

int main() {
  bool ok = true;

  const bool helpOk = testHelpRequest();
  std::cout << "testHelpRequest: " << (helpOk ? "ok" : "failed") << '\n';
  ok = ok && helpOk;

  const bool backendOk = testBackendAndPositionals();
  std::cout << "testBackendAndPositionals: " << (backendOk ? "ok" : "failed") << '\n';
  ok = ok && backendOk;

  const bool oddPositionalsOk = testRejectOddPositionals();
  std::cout << "testRejectOddPositionals: " << (oddPositionalsOk ? "ok" : "failed") << '\n';
  ok = ok && oddPositionalsOk;

  const bool unknownOptionOk = testRejectUnknownOption();
  std::cout << "testRejectUnknownOption: " << (unknownOptionOk ? "ok" : "failed") << '\n';
  ok = ok && unknownOptionOk;

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
