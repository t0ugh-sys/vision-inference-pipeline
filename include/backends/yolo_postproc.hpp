#pragma once

#include "../postproc_interface.hpp"

class YoloPostprocessor : public IPostprocessor {
 public:
  YoloPostprocessor(YoloVersion version, PostprocessOptions options = {});

  DetectionResult postprocess(
      const InferenceOutput& output,
      const RgbImage& modelInput,
      int originalWidth,
      int originalHeight,
      int64_t pts) override;

  std::string name() const override;

 private:
  DetectionResult postprocessDenseTensor(
      const InferenceTensor& tensor,
      const RgbImage& modelInput,
      int originalWidth,
      int originalHeight,
      int64_t pts) const;
  DetectionResult postprocessBranchOutputs(
      const InferenceOutput& output,
      const RgbImage& modelInput,
      int originalWidth,
      int originalHeight,
      int64_t pts) const;
  DetectionResult postprocessYolo26E2E(
      const InferenceTensor& tensor,
      const RgbImage& modelInput,
      int originalWidth,
      int originalHeight,
      int64_t pts) const;

  const std::vector<std::string>& labelsForClassCount(int classCount) const;
  ModelOutputLayout inferLayout(const InferenceOutput& output) const;
  std::vector<float> valuesAsFloat(const InferenceTensor& tensor) const;
  int classChannelCount(const InferenceOutput& output) const;

  static float computeIoU(const BoundingBox& a, const BoundingBox& b);
  static std::vector<BoundingBox> nms(std::vector<BoundingBox>& boxes, float iouThreshold);
  static void mapBoxesToOriginal(
      std::vector<BoundingBox>& boxes,
      const RgbImage& modelInput,
      int originalWidth,
      int originalHeight);
  static const char* layoutToString(ModelOutputLayout layout);

  YoloVersion version_;
  PostprocessOptions options_;
  mutable std::vector<std::string> cachedGeneratedLabels_;
  mutable bool autoLayoutLogged_ = false;
  mutable bool denseTensorStatsLogged_ = false;
  mutable bool branchTimingLogged_ = false;
  mutable bool flatExperimentalLogged_ = false;
  mutable bool denseOrientationCached_ = false;
  mutable int denseOrientationCachedProposals_ = 0;
  mutable int denseOrientationCachedAttributes_ = 0;
  mutable int denseOrientationCachedClassOffset_ = 0;
  mutable int denseOrientationCachedClassCount_ = 0;
  mutable bool denseOrientationCachedHasObjectness_ = false;
  mutable bool denseOrientationUseTransposed_ = false;
};
