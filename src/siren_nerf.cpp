#include "siren_nerf.h"

SirenLayer::SirenLayer(int64_t dim_in, int64_t dim_out, 
                       bool is_first, float w0, bool use_bias, float c)
    : dim_in_(dim_in), is_first_(is_first), w0_(w0) {
    // Initialize weight and bias
    weight_ = register_parameter("weight", torch::zeros({dim_out, dim_in}));
    float w_std = is_first_ ? (1.0f / dim_in_) : (std::sqrt(c / dim_in_) / w0_);
    torch::nn::init::uniform_(weight_, -w_std, w_std);
    if (use_bias) {
        bias_ = register_parameter("bias", torch::zeros(dim_out));
        torch::nn::init::uniform_(bias_, -w_std, w_std);
    }
}

torch::Tensor SirenLayer::forward(const torch::Tensor &x) {
    auto out = torch::nn::functional::linear(x, weight_, bias_);
    return torch::sin(w0_ * out);
}

torch::Tensor SirenLayer::get_weights() {
    return weight_;
}


SirenNeRF::SirenNeRF(torch::Device device, int W, int D): device_(device) {
    // Ensure D is at least 2
    D = std::max(D, 2);

    // Create position encoder SIREN layers
    pos_siren_ = std::make_shared<SirenLayer>(1, 64, true, 120);
    register_module("pos_siren", pos_siren_);
    pos_siren_->to(device_);

    // Create view direction encoder SIREN layers
    view_siren_ = std::make_shared<SirenLayer>(1, 32, true, 20);
    register_module("view_siren", view_siren_);
    view_siren_->to(device_);

    // Create position encoding network
    nerf_net_ = torch::nn::Sequential();
    nerf_net_->push_back(SirenLayer(64*3, W));
    for (int i = 0; i < D; i++) {
        nerf_net_->push_back(SirenLayer(W, W));
    }
    nerf_net_->to(device_);
    register_module("nerf_net", nerf_net_);

    // Create sigma head (single layer)
    sigma_head_ = torch::nn::Sequential();
    sigma_head_->push_back(LinearLayer(W, W));
    sigma_head_->push_back(LinearLayer(W, 1));
    sigma_head_->push_back(torch::nn::ReLU());
    sigma_head_->to(device_);
    register_module("sigma_head", sigma_head_);

    // Create rgb head
    rgb_head_ = torch::nn::Sequential();
    rgb_head_->push_back(SirenLayer(W + 32*3, W));
    rgb_head_->push_back(LinearLayer(W, W));
    rgb_head_->push_back(LinearLayer(W, 3));
    rgb_head_->push_back(torch::nn::Sigmoid());
    rgb_head_->to(device_);
    register_module("rgb_head", rgb_head_);

    this->to(device_);
}

NeRFOutput SirenNeRF::forward(const torch::Tensor &pts, const torch::Tensor &view_dirs) {
    // Position encoding
    auto pos_unsqueezed = pts.unsqueeze(-1);
    auto pos_siren_out = pos_siren_->forward(pos_unsqueezed);
    auto pos_features = pos_siren_out.flatten(1);
    
    // View direction encoding
    auto view_unsqueezed = view_dirs.unsqueeze(-1);
    auto view_siren_out = view_siren_->forward(view_unsqueezed);
    auto view_features = view_siren_out.flatten(1);
    
    // Process through nerf_net
    auto nerf_features = nerf_net_->forward(pos_features);
    
    // Concatenate processed features with view direction for RGB
    torch::Tensor rgb_features = torch::cat({nerf_features, view_features}, -1);
    
    // Generate outputs
    NeRFOutput output;
    output.sigma = sigma_head_->forward(nerf_features);
    output.rgb = rgb_head_->forward(rgb_features);
    
    return output;
}

torch::Tensor SirenNeRF::siren_diversity_loss() {
   
    auto pos_weights = pos_siren_->get_weights().flatten();
    auto view_weights = view_siren_->get_weights().flatten();
    
    // Compute standard deviation of weights
    auto pos_std = pos_weights.std();
    auto view_std = view_weights.std();
    
    return torch::exp(-pos_std*pos_std / 0.2) + \
           torch::exp(-view_std*view_std / 0.2);
}
