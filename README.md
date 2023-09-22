# arm64-docker-pytorch-tensorflow

Dockerfile of a Deep Learning development environment with PyTorch and Tensorflow for arm64 architecture, especially Apple Silicon (M1/M2) Macs.
This Docker image is configured for deep learning tasks on CPUs and is optimized for use on Apple Silicon (M1/M2) Macs. 

Please note that this image is based on the Ubuntu operating system, which, at least currently, does not support Apple's Metal Performance Shaders (MPS). MPS is a technology designed specifically for Apple's hardware and software ecosystem, and it relies on Apple's Metal framework and GPU architecture. Docker containers on Linux do not natively support Metal Performance Shaders because they are not designed to run macOS or iOS applications, which is where Metal and MPS are typically used.

As per the official PyTorch guide, MPS is supported exclusively on the latest macOS, and its functionality is not extended to Ubuntu-based systems. This means that even when running this PyTorch Docker image on the latest macOS, the image itself cannot harness the power of Apple's custom GPU architecture.

For GPU-accelerated deep learning tasks on Apple Silicon Macs, the only available current option is to use the macOS environment, which provides the necessary support for MPS and GPU acceleration.

* Docker image: https://hub.docker.com/r/sonoisa/deep-learning-coding
    * ```docker pull sonoisa/deep-learning-coding:pytorch1.6.0_tensorflow2.3.0```
    * ```docker pull sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1```

This Dockerfile is a merged version of the Dockerfile for PyTorch and the Dockerfile for TensorFlow described in the following site.
https://community.arm.com/developer/tools-software/tools/b/tools-software-ides-blog/posts/aarch64-docker-images-for-pytorch-and-tensorflow


## How to build

### To build on 16GB M1/M2 MacBook Pro

- Make sure to allocate 12GB of memory and 4 cpu to Docker.
- The build may fail due to lack of memory, so restart the Docker service itself and run the build immediately after that.

```
$ ./build.sh --build-type full --jobs 2
```


### To build on 64GB M1/M2 Max MacBook Pro

- Make sure to allocate 48GB of memory and 8 cpu to Docker.

```
$ ./build.sh --build-type full --jobs 8 --bazel_memory_limit 24576
```
