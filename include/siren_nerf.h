#ifndef SIREN_NERF_H_
#define SIREN_NERF_H_

#include "nerf_model.h"

class SirenLayer : public torch::nn::Module {
public:
    explicit SirenLayer(int64_t dim_in, 
                       int64_t dim_out, 
                       bool is_first = false, 
                       float w0, 
                       bool use_bias = true,
                       float c = 6.0f);

    torch::Tensor forward(const torch::Tensor &x);
    torch::Tensor get_weights();

private:
    int64_t dim_in_;
    bool is_first_;
    float w0_;
    torch::Tensor weight_;
    torch::Tensor bias_;
};

class SirenNeRF : public NeRFModel {
public:
  SirenNeRF(torch::Device device = torch::kCPU, int W = 128, int D = 4);

  NeRFOutput forward(const torch::Tensor &pts, const torch::Tensor &view_dirs) override;
  
  // Calculate diversity loss for SIREN position encoder weights
  torch::Tensor siren_diversity_loss();

private:
  std::shared_ptr<SirenLayer> pos_siren_;
  std::shared_ptr<SirenLayer> view_siren_;
  torch::nn::Sequential nerf_net_;
  torch::nn::Sequential sigma_head_;
  torch::nn::Sequential rgb_head_;
  const torch::Device device_;
};

#endif // SIREN_NERF_H_
