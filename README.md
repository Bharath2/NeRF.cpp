# NeRF.cpp
NeRF implementation with Siren layers in C++.

<p align="center">
  <img src="./output/animation.gif" alt="NeRF Animation" loop="infinite"/>
</p>

## Requirements
- CUDA
- LibTorch
- nlohmann_json

## Building and Running

Build the project with cmake.

```
mkdir build
cd build
cmake ..
make
```
Run the executable with input and output paths.

```
./NeRF.cpp /path/to/data/lego /path/to/output
```

## References

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al., ECCV 2020)
- [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) (Sitzmann et al., NeurIPS 2020)
- [cNeRF project](https://github.com/rafaelanderka/cNeRF) by rafaelanderka.
