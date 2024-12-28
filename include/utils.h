#ifndef UTILS_H_
#define UTILS_H_

#include <filesystem>
#include <string>
#include <torch/torch.h>

#include "renderer.h"


struct Dataset {
    torch::Tensor images;
    torch::Tensor poses;
    float focal;
    int len;
};

Dataset load_dataset(const std::string& json_path, int target_width = 200);

// Initialization functions
void set_seed(int seed);
torch::Device get_device();
bool parse_arguments(int argc, char *argv[], std::filesystem::path &data_path,
                     std::filesystem::path &output_path);

// Image loading/saving functions
torch::Tensor load_image(const std::string& image_path);
void save_image(const torch::Tensor &tensor, const std::filesystem::path &file_path);

// Rendering helper functions
void render_views(const NeRFRenderer &renderer, const std::string &prefix, 
                  int H, int W, int N_frames,
                  const std::filesystem::path &output_folder,
                  float radius = 4.0f,
                  float start_distance = 2.0f,
                  float end_distance = 5.0f, int N_samples = 256);

// Transformation and pose functions
torch::Tensor create_spherical_pose(float azimuth, float elevation,
                                    float radius);
torch::Tensor create_translation_matrix(float t);
torch::Tensor create_phi_rotation_matrix(float phi);
torch::Tensor create_theta_rotation_matrix(float theta);

// Checkpoint functions
void save_checkpoint(const std::filesystem::path &path,
                    const torch::nn::Module &model,
                    const torch::optim::Adam &optimizer,
                    int epoch);

void load_checkpoint(const std::filesystem::path &path,
                    torch::nn::Module &model,
                    torch::optim::Adam &optimizer,
                    int &epoch);

#endif // UTILS_H_
