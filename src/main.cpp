#include "app_config.hpp"
#include "pipeline_runner.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void setGpuEnvironment(int gpuId) {
  if (gpuId < 0) {
    return;
  }

#ifdef _WIN32
  if (_putenv_s("CUDA_DEVICE", std::to_string(gpuId).c_str()) != 0) {
    throw std::runtime_error("Failed to set CUDA_DEVICE environment variable");
  }
#else
  if (setenv("CUDA_DEVICE", std::to_string(gpuId).c_str(), 1) != 0) {
    throw std::runtime_error("Failed to set CUDA_DEVICE environment variable");
  }
#endif
}

}  // namespace

int main(int argc, char* argv[]) {
  const ParseResult parseResult = parseAppConfig(argc, argv);

  if (parseResult.status == ParseStatus::kHelp) {
    std::cout << parseResult.message;
    return 0;
  }

  if (parseResult.status == ParseStatus::kError) {
    std::cerr << parseResult.message;
    return 1;
  }

  try {
    validateAppConfig(parseResult.config);
    setGpuEnvironment(parseResult.config.gpuId);
    runPipeline(parseResult.config);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "\n[ERROR] Pipeline failed: " << error.what() << '\n';
    return 1;
  }
}
