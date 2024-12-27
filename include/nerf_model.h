#ifndef NERF_MODEL_H_
#define NERF_MODEL_H_

#include <torch/torch.h>
#include <cmath>
#include <memory>
#include <vector>
#include <iostream>

struct NeRFOutput {
  torch::Tensor rgb;
  torch::Tensor sigma;
};

class NeRFModel : public torch::nn::Module {
public:
  virtual ~NeRFModel() = default;
  
  // Pure virtual function that derived classes must implement
  virtual NeRFOutput forward(const torch::Tensor &pts, const torch::Tensor &view_dirs) = 0;
};


class LinearLayer : public torch::nn::Module {
public:
    explicit LinearLayer(int64_t dim_in,
                        int64_t dim_out,
                        bool use_bias = true)
        : dim_in_(dim_in), dim_out_(dim_out), use_bias_(use_bias) {
        // Initialize weight and bias
        weight_ = register_parameter("weight", torch::zeros({dim_out, dim_in}));
        torch::nn::init::xavier_normal_(weight_);
        if (use_bias) {
            bias_ = register_parameter("bias", torch::zeros(dim_out));
            torch::nn::init::constant_(bias_, 0.1f);
        }
    }

    torch::Tensor forward(const torch::Tensor &x) {
        return torch::nn::functional::linear(x, weight_, bias_);
    }

private:
    int64_t dim_in_;
    int64_t dim_out_;
    bool use_bias_;
    torch::Tensor weight_;
    torch::Tensor bias_;
};

#endif // NERF_MODEL_H_ 