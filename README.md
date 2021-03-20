# arm64-docker-pytorch-tensorflow

Dockerfile of a Deep Learning development environment with PyTorch and Tensorflow for arm64 architecture, especially Applie Silicon (M1) Macs.

* Docker image: https://hub.docker.com/repository/docker/sonoisa/deep-learning-coding

This Dockerfile is a merged version of the Dockerfile for PyTorch and the Dockerfile for TensorFlow described in the following site.
https://community.arm.com/developer/tools-software/tools/b/tools-software-ides-blog/posts/aarch64-docker-images-for-pytorch-and-tensorflow


## How to build

Notes
- Make sure to allocate 12GB of memory to Docker.
- The build may fail due to lack of memory, so restart the Docker service itself and run the build immediately after that.

```
$ ./build.sh --build-type full --jobs 2
```

