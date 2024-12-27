#ifndef RENDERER_H_
#define RENDERER_H_

#include <torch/torch.h>
#include "siren_nerf.h"

enum class SampleStrategy {
  UNIFORM,
  RANDOM,
  PROPOSAL
};

struct RenderOutput {
  torch::Tensor rgb;
  torch::Tensor depth;
};

class NeRFRenderer {
public:
  NeRFRenderer(NeRFModel &model, float focal, torch::Device device);

  // Full render with sampling options
  RenderOutput render_rays(int H, int W,
                    const torch::Tensor &pose, 
                    SampleStrategy sample_strategy = SampleStrategy::UNIFORM,
                    float start_distance = 2.0f, float end_distance = 6.0f,
                    int n_samples = 96, int batch_size = 320000) const;

  // Direct render with provided points
  RenderOutput volume_render(int H, int W,
                           const torch::Tensor &rays_o,
                           const torch::Tensor &rays_d,
                           const torch::Tensor &z_vals,
                           int batch_size = 320000) const;

private:
  NeRFModel &model_;
  const torch::Device device_;
  float focal_;
};

#endif // RENDERER_H_
