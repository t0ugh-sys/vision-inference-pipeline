#pragma once

#include "decoder_interface.hpp"
#include "encoder_interface.hpp"
#include "infer_interface.hpp"
#include "postproc_interface.hpp"
#include "preproc_interface.hpp"

#include <string>

std::string toString(DecoderBackendType type);
std::string toString(PreprocBackendType type);
std::string toString(InferBackendType type);
std::string toString(PostprocBackendType type);
std::string toString(EncoderBackendType type);

bool isCompiledIn(DecoderBackendType type);
bool isCompiledIn(PreprocBackendType type);
bool isCompiledIn(InferBackendType type);
bool isCompiledIn(PostprocBackendType type);
bool isCompiledIn(EncoderBackendType type);

std::string availableDecoderBackends();
std::string availablePreprocBackends();
std::string availableInferBackends();
std::string availablePostprocBackends();
std::string availableEncoderBackends();
