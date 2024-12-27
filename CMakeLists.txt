cmake_minimum_required(VERSION 3.20)

project(NeRF.cpp LANGUAGES CXX)

# Configure CUDA
find_package(CUDA)
find_library(LIBNVTOOLSEXT
    NAMES nvToolsExt
    HINTS
        ENV CUDA_TOOLKIT_ROOT_DIR
        ENV CONDA_PREFIX
    PATH_SUFFIXES lib64 lib
)

# Configure libtorch, and nlohmann_json
list(APPEND CMAKE_PREFIX_PATH "libtorch")
find_package(Torch REQUIRED)
find_package(nlohmann_json REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(NeRF.cpp src/main.cpp src/siren_nerf.cpp src/renderer.cpp src/utils.cpp)
target_include_directories(NeRF.cpp PUBLIC include ${JSON_INCLUDE_DIRS})
target_link_libraries(NeRF.cpp ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} nlohmann_json::nlohmann_json)
