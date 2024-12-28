#include "siren_nerf.h"
#include "renderer.h"
#include "utils.h"
#include <fstream>
#include <random>

//Training Loop options
constexpr int n_iters = 6000;
constexpr int log_freq = 50;
constexpr int plot_freq = 100;
constexpr int n_preview_frames = 5;
constexpr int n_final_frames = 30;

//Render options
constexpr int batch_size = 1280000;
constexpr float start_distance = 2.0f;
constexpr float end_distance = 8.0f;
constexpr int n_samples = 96; //no. of samples along each Ray

int main(int argc, char *argv[]) {
  // Parse command-line arguments
  std::filesystem::path data_path;
  std::filesystem::path output_path;
  if (!parse_arguments(argc, argv, data_path, output_path)) {
    return 1;
  }
  
  // Determine device for computation
  torch::Device device = get_device();

  // Load data: images, poses, and focal length
  std::filesystem::path transforms_path = data_path / "transforms.json";
  Dataset dataset = load_dataset(transforms_path.string(), 120); //images will be resized to 120x120

  // Display information about the loaded data
  std::cout << "Images: " << dataset.images.sizes() << std::endl;
  std::cout << "Focal length: " << dataset.focal << std::endl;
  std::cout << "dataset length: " << dataset.len << std::endl;

  // Create model and optimizer
  SirenNeRF model(device);
  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

  // Create renderer with main model
  NeRFRenderer renderer(model, dataset.focal, device);

  // Set up uniform distribution for random image selection
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(0, dataset.len - 1);

  // Train the NeRF model
  for (int i = 0; i < n_iters; i++) {
    // Sample a random image and its corresponding pose
    int img_i = dist(rng);
    auto target = dataset.images[img_i].to(device);
    auto pose = dataset.poses[img_i].to(device);

    // Zero gradients
    optimizer.zero_grad();

    // Render and Compute losses
    auto out = renderer.render_rays(target.size(0), target.size(1), 
                                     pose, SampleStrategy::UNIFORM,
                                     start_distance, end_distance,
                                     n_samples, batch_size);

    auto rgb_loss = torch::mse_loss(out.rgb, target);
    
    // Get diversity loss from position encoder
    auto diversity_loss = model.siren_diversity_loss();
    auto total_loss = rgb_loss + 0.01 * diversity_loss;

    // Backward and optimize
    total_loss.backward();
    optimizer.step();

    // Log progress every 10 iterations
    if (i % log_freq == 0) {
      std::cout << "Iteration: " << i + 1 
                << " Loss: " << rgb_loss.item<float>()
                << std::endl;
    }


    // Save preview images every 100 iterations
    if (i % plot_freq == 0) {
      torch::NoGradGuard no_grad;
      render_views(renderer, std::to_string(i), 240, 240, //saves 240x240 images
                   n_preview_frames, output_path, 2.1f,
                   0.8f, 3.2f);
      auto checkpoint_path = output_path / "checkpoint.pt";
      save_checkpoint(checkpoint_path, model, optimizer, i);
    }
  }

  // Save final checkpoint
  save_checkpoint(output_path / "checkpoint.pt", model, optimizer, n_iters);

  std::cout << "Done" << std::endl;

  // Generate the final render using the trained model
  torch::NoGradGuard no_grad;
  render_views(renderer, "final", 240, 240, 
               n_final_frames, output_path, 2.1f,
               0.8f, 3.2f);

  return 0;
}
