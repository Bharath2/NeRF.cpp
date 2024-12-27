#include "renderer.h"
#include "utils.h"

using namespace torch::indexing;

NeRFRenderer::NeRFRenderer(NeRFModel &model, float focal, torch::Device device)
    : model_(model), device_(device), focal_(focal) {}

RenderOutput NeRFRenderer::render_rays(int H, int W,
                                    const torch::Tensor &pose,
                                    SampleStrategy sample_strategy,
                                    float start_distance, float end_distance, 
                                    int n_samples, int batch_size) const {
  // Create pixel coordinate grids
  auto x_coords = torch::arange(W, torch::dtype(torch::kFloat32)).to(device_);
  auto y_coords = torch::arange(H, torch::dtype(torch::kFloat32)).to(device_);
  auto grid = torch::meshgrid({x_coords, y_coords}, "xy");
  auto x_grid = grid[0];  // [H, W]
  auto y_grid = grid[1];  // [H, W]

  // Convert pixel coordinates to normalized image plane coordinates
  // Center is at (W/2, H/2), scaled by focal length
  auto dirs = torch::stack({
      (x_grid - W * 0.5f) / focal_,      // x component
      -(y_grid - H * 0.5f) / focal_,     // y component (negative because y points down)
      -torch::ones_like(x_grid)           // z component (negative because camera looks down -z)
  }, -1);  // [H, W, 3]

  // Transform ray directions from camera to world space
  auto camera_to_world = pose.index({Slice(0, 3), Slice(0, 3)});  // [3, 3]
  auto rays_d = torch::sum(
      dirs.index({"...", None, Slice()}) * camera_to_world,  // [H, W, 1, 3] * [3, 3]
      -1  // Sum along last dimension
  );  // [H, W, 3]

  // Get ray origins from camera position (last column of pose matrix)
  auto rays_o = pose.index({Slice(0, 3), -1}).expand(rays_d.sizes());    // [H, W, 3]

  // Create sampling points along each ray
  auto z_vals = torch::linspace(start_distance, end_distance, n_samples, device_)
                    .reshape({1, 1, n_samples})      // [1, 1, n_samples]
                    .expand({H, W, n_samples})     // [H, W, n_samples]
                    .clone();

  // Apply sampling strategy
  if (sample_strategy == SampleStrategy::RANDOM) {
    auto z_vals_new = z_vals + torch::rand({H, W, n_samples}, device_) *
              (start_distance - end_distance) / n_samples;

    z_vals = torch::cat({z_vals, z_vals_new}, -1);
  }

  // // Apply sampling strategy
  // if (sample_strategy == SampleStrategy::PROPOSAL) {
  //   auto z_vals_new = z_vals + torch::rand({H, W, n_samples}, device_) *
  //             (start_distance - end_distance) / n_samples;

  //   z_vals = torch::cat({z_vals, z_vals_new}, -1);
  // }

  return volume_render(H, W, rays_o, rays_d, z_vals, batch_size);
}

RenderOutput NeRFRenderer::volume_render(int H, int W,
                                      const torch::Tensor &rays_o,
                                      const torch::Tensor &rays_d,
                                      const torch::Tensor &z_vals,
                                      int batch_size) const {
  // Calculate points along each ray and flatten [H, W, N] to [-1, N]
  auto pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1);
  auto pts_flat = pts.view({-1, 3});
  
  // Get viewing directions and normalize them
  auto rays_norm = torch::norm(rays_d, 2, -1, true);
  auto view_dirs = rays_d / (rays_norm + 1e-8);
  auto view_dirs_flat = view_dirs.unsqueeze(-2).expand(pts.sizes()).reshape({-1, 3});

  // Batch-process points
  int n_pts = pts_flat.size(0);
  std::vector<torch::Tensor> rgb_outputs;
  std::vector<torch::Tensor> sigma_outputs;
  std::unique_ptr<NeRFOutput> batch_output;
  for (int i = 0; i < n_pts; i += batch_size) {
    if(i != 0){
      rgb_outputs.push_back(batch_output->rgb);
      sigma_outputs.push_back(batch_output->sigma);
    }
    auto end_idx = std::min<int64_t>(i + batch_size, n_pts);
    auto pts_batch = pts_flat.slice(0, i, end_idx);
    auto view_dirs_batch = view_dirs_flat.slice(0, i, end_idx);
    batch_output = std::make_unique<NeRFOutput>(model_.forward(pts_batch, view_dirs_batch));
  }

  // Combine outputs and reshape
  auto rgb = torch::cat({torch::stack(rgb_outputs).view({-1, 3}), 
                         batch_output->rgb}).view({H, W, z_vals.size(-1), 3});
  auto sigma = torch::cat({torch::stack(sigma_outputs).view({-1, 1}),
                           batch_output->sigma}).view({H, W, z_vals.size(-1)});

  // Calculate distances between adjacent sampling points along rays
  auto dists = torch::cat({
      // Distances between consecutive points
      z_vals.index({"...", Slice(1, None)}) - z_vals.index({"...", Slice(None, -1)}),
      // Add large value at the end for last segment
      torch::full({1}, 1e10, device_).expand({H, W, 1})
  }, -1);

  // Calculate alpha values (opacity) from density (sigma) and distances
  auto alpha = 1.0 - torch::exp(-sigma * dists);

  // Calculate transmittance (probability of light reaching each point)
  auto transmittance = torch::cumprod(1.0 - alpha + 1e-10, -1);

  // Calculate final weights for volume rendering
  auto weights = alpha * torch::cat({
      torch::ones({H, W, 1}, device_),  // Weight for first point
      transmittance.index({"...", Slice(None, -1)})  // Weights for remaining points
  }, -1);

  RenderOutput out;
  out.rgb = torch::sum(weights.unsqueeze(-1) * rgb, -2);
  out.depth = torch::sum(weights * z_vals, -1);
  return out;
}