#include "utils.h"
#include "renderer.h"
#include "nlohmann/json.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>


torch::Device get_device() {
  static torch::Device cached_device = []() {
    if (torch::cuda::is_available()) {
      std::cout << "Using CUDA device" << std::endl;
      return torch::kCUDA;
    }
    std::cout << "Using CPU device" << std::endl;
    return torch::kCPU;
  }();
  return cached_device;
}

bool parse_arguments(int argc, char *argv[], std::filesystem::path &data_path,
                     std::filesystem::path &output_path) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data_path> <output_path>"
              << std::endl;
    return false;
  }

  data_path = argv[1];
  output_path = argv[2];
  return true;
}


torch::Tensor load_image(const std::string& image_path) {
    int width, height;
    uint8_t* data = stbi_load(image_path.c_str(), &width, &height, nullptr, 3);
    if (!data) {
        std::cerr << "Failed to load: " << image_path << std::endl;
        return torch::empty({0});
    }

    auto tensor = torch::from_blob(data, {height, width, 3}, torch::kUInt8).clone();
    tensor = tensor.to(torch::kFloat32)/(255.0);
    stbi_image_free(data);
    return tensor;
}

void save_image(const torch::Tensor &tensor, const std::filesystem::path &file_path) {
    // Assuming the input tensor is a 3-channel (HxWx3) image in the range [0, 1]
    auto height = tensor.size(0);
    auto width = tensor.size(1);
    auto tensor_normalized = tensor.mul(255)
                                 .clamp(0, 255)
                                 .to(torch::kU8)
                                 .to(torch::kCPU)
                                 .flatten()
                                 .contiguous();

    if (stbi_write_jpg(file_path.string().c_str(), width, height, 3,
                      static_cast<uint8_t*>(tensor_normalized.data_ptr()), 100) == 0) {
        std::cerr << "Failed to save: " << file_path << std::endl;
    }
}


// Read camera poses  and Images from transforms.json
Dataset load_dataset(const std::string& json_path, int target_width) {
    std::vector<torch::Tensor> poses;
    std::vector<torch::Tensor> images;
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open transforms.json");
    }

    nlohmann::json data;
    file >> data;

    float camera_angle_x = data["camera_angle_x"];
    float focal = 0.5f * static_cast<float>(target_width) / std::tan(0.5f * camera_angle_x);
    
    std::filesystem::path json_dir = std::filesystem::path(json_path).parent_path();
    
    for (const auto& frame : data["frames"]) {
        // Load image
        std::string image_path = json_dir / (frame["file_path"].get<std::string>() + ".png");
        auto image = load_image(image_path);
        images.push_back(image);
        
        // Load pose
        std::array<float, 16> matrix_data;
        int idx = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                matrix_data[idx++] = frame["transform_matrix"][i][j];
            }
        }
        
        auto pose = torch::from_blob(matrix_data.data(), {4, 4}, torch::kFloat32).clone();
        poses.push_back(pose);
    }

    // Stack tensors
    auto images_tensor = torch::stack(images).to(torch::kCPU);
    auto poses_tensor = torch::stack(poses).to(torch::kCPU);
    
    // Calculate height while maintaining aspect ratio
    float aspect_ratio = static_cast<float>(images_tensor.size(2)) / images_tensor.size(1);
    int target_height = static_cast<int>(target_width * aspect_ratio);
    
    // Resize stacked images if needed
    if (images_tensor.size(1) != target_height || images_tensor.size(2) != target_width) {
        images_tensor = torch::nn::functional::interpolate(
            images_tensor.permute({0, 3, 1, 2}),  // NHWC -> NCHW
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_height, target_width})
                .mode(torch::kBilinear)
                .align_corners(true)
        ).permute({0, 2, 3, 1});  // NCHW -> NHWC
    }
    
    int len = images_tensor.size(0);

    return Dataset{images_tensor, poses_tensor, focal, len};
}


void render_views(const NeRFRenderer &renderer, 
                  const std::string &prefix, int H, int W,
                  int num_frames,
                  const std::filesystem::path &output_folder,
                  float radius, float start_distance,
                  float end_distance, int n_samples) {
  float elevation = -30.0f;
  auto device = get_device();

  std::cout << "Saving " << num_frames << " sample views..." << std::endl;

  for (int i = 0; i < num_frames; i++) {
    float azimuth = static_cast<float>(i) * 360.0f / num_frames;
    auto pose = create_spherical_pose(azimuth, elevation, radius).to(device);

    auto rendered_output =
        renderer.render_rays(H, W, pose, SampleStrategy::UNIFORM, 
                           start_distance, end_distance, n_samples);

    std::string file_path = output_folder / ("frame_" + prefix + "_" + std::to_string(i) + ".png");
    save_image(rendered_output.rgb, file_path);

    std::string depth_file_path = output_folder / ("frame_depth_" + prefix + "_" + std::to_string(i) + ".png");
    auto depth_min = rendered_output.depth.min();
    auto depth_max = rendered_output.depth.max();
    auto depth_normalized = (rendered_output.depth - depth_min) / (depth_max - depth_min);
    auto depth_rgb = torch::zeros({depth_normalized.size(0), depth_normalized.size(1), 3}, device);
    // Set all channels to the same value for grayscale
    depth_rgb.index_put_({torch::indexing::Ellipsis, 0}, depth_normalized);
    depth_rgb.index_put_({torch::indexing::Ellipsis, 1}, depth_normalized);
    depth_rgb.index_put_({torch::indexing::Ellipsis, 2}, depth_normalized);
    save_image(depth_rgb, depth_file_path);
  }
}

torch::Tensor create_spherical_pose(float azimuth, float elevation,
                                    float radius) {
  float phi = elevation * (M_PI / 180.0f);
  float theta = azimuth * (M_PI / 180.0f);

  torch::Tensor c2w = create_translation_matrix(radius);
  c2w = create_phi_rotation_matrix(phi).matmul(c2w);
  c2w = create_theta_rotation_matrix(theta).matmul(c2w);
  c2w = torch::tensor({{-1.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 1.0f, 0.0f},
                       {0.0f, 1.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 1.0f}})
            .matmul(c2w);

  return c2w;
}

torch::Tensor create_translation_matrix(float t) {
  torch::Tensor t_mat = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                                       {0.0f, 1.0f, 0.0f, 0.0f},
                                       {0.0f, 0.0f, 1.0f, t},
                                       {0.0f, 0.0f, 0.0f, 1.0f}});
  return t_mat;
}

torch::Tensor create_phi_rotation_matrix(float phi) {
  torch::Tensor phi_mat =
      torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                     {0.0f, std::cos(phi), -std::sin(phi), 0.0f},
                     {0.0f, std::sin(phi), std::cos(phi), 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}});
  return phi_mat;
}

torch::Tensor create_theta_rotation_matrix(float theta) {
  torch::Tensor theta_mat =
      torch::tensor({{std::cos(theta), 0.0f, -std::sin(theta), 0.0f},
                     {0.0f, 1.0f, 0.0f, 0.0f},
                     {std::sin(theta), 0.0f, std::cos(theta), 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}});
  return theta_mat;
}

void save_checkpoint(const std::filesystem::path& path,
                    const torch::nn::Module& model,
                    const torch::optim::Adam& optimizer,
                    int epoch) {
    torch::serialize::OutputArchive archive;
    
    // Save model
    model.save(archive);
    // Save epoch
    archive.write("epoch", epoch);  
    archive.save_to(path.string());
    std::cout << "Model Weights saved"<< std::endl;
}

void load_checkpoint(const std::filesystem::path& path,
                    torch::nn::Module& model,
                    torch::optim::Adam& optimizer,
                    int& epoch) {
    torch::serialize::InputArchive archive;
    archive.load_from(path.string());
    // Load model
    model.load(archive);
    // Load epoch
    c10::IValue epoch_val;  // Create an IValue to store the read value
    archive.read("epoch", epoch_val);  // Read into the IValue
    epoch = epoch_val.toInt();
    std::cout << "Checkpoint loaded from " << path << " at epoch " << epoch << std::endl;
}