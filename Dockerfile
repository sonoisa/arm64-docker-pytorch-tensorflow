# *******************************************************************************
# Copyright 2020-2021 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

# ========
# Stage 1: Base image including OS and key packages
# ========
ARG njobs
ARG bazel_mem
ARG default_py_version=3.8

FROM ubuntu:20.04 AS deep-learning-base
ARG default_py_version
ENV PY_VERSION="${default_py_version}"

RUN if ! [ "$(arch)" = "aarch64" ] ; then exit 1; fi

#Install core OS packages
RUN apt-get -y update && \
    apt-get -y install software-properties-common --no-install-recommends && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get -y install --no-install-recommends \
      accountsservice \
      apport \
      at \
      autoconf \
      bc \
      build-essential \
      cmake \
      cpufrequtils \
      curl \
      ethtool \
      g++-9 \
      g++-10 \
      gcc-7 \
      gcc-9 \
      gcc-10 \
      gettext-base \
      gfortran-9 \
      gfortran-10 \
      git \
      iproute2 \
      iputils-ping \
      lxd \
      libbz2-dev \
      libc++-dev \
      libcgal-dev \
      libffi-dev \
      libfreetype6-dev \
      libhdf5-dev \
      libjpeg-dev \
      liblzma-dev \
      libncurses5-dev \
      libncursesw5-dev \
      libpng-dev \
      libprotoc-dev \
      libreadline-dev \
      libsox-dev \
      libsox-fmt-all \
      libssl-dev \
      libsqlite3-dev \
      libxml2-dev \
      libxslt-dev \
      locales \
      lsb-release \
      lvm2 \
      moreutils \
      net-tools \
      open-iscsi \
      openjdk-8-jdk \
      openssl \
      pciutils \
      pkg-config \
      policykit-1 \
      python${PY_VERSION} \
      python${PY_VERSION}-dev \
      python${PY_VERSION}-distutils \
      python${PY_VERSION}-venv \
      python3-pip \
      python-openssl \
      protobuf-compiler \
      rsync \
      rsyslog \
      snapd \
      scons \
      sox \
      ssh \
      sudo \
      time \
      udev \
      unzip \
      ufw \
      uuid-runtime \
      vim \
      wget \
      xz-utils \
      zip \
      zlib1g-dev \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# Make gcc 9 the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1 --slave /usr/bin/g++ g++ /usr/bin/g++-10 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# DOCKER_USER for the Docker user
ENV DOCKER_USER=ubuntu

# Setup default user
RUN useradd --create-home -s /bin/bash -m $DOCKER_USER && echo "$DOCKER_USER:Portland" | chpasswd && adduser $DOCKER_USER sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Import profile for bash
COPY bash_profile /home/$DOCKER_USER/.bash_profile
RUN chown $DOCKER_USER:$DOCKER_USER /home/$DOCKER_USER/.bash_profile
COPY patches/welcome.txt /home/$DOCKER_USER/.
RUN echo '[ ! -z "$TERM" -a -r /home/$DOCKER_USER/welcome.txt ] && cat /home/$DOCKER_USER/welcome.txt' >> /etc/bash.bashrc


# ========
# Stage 2: augment the base image with some essential libs
# ========
FROM deep-learning-base AS deep-learning-libs
ARG njobs
ARG cpu
ARG arch
ARG blas_cpu
ARG blas_ncores
ARG acl_arch
ARG pt_onednn_opt
ARG tf_onednn_opt

ENV NP_MAKE="${njobs}" \
    CPU="${cpu}" \
    ARCH="${arch}" \
    BLAS_CPU="${blas_cpu}" \
    BLAS_NCORES="${blas_ncores}" \
    ACL_ARCH="${acl_arch}" \
    PT_ONEDNN_BUILD="${pt_onednn_opt}" \
    TF_ONEDNN_BUILD="${tf_onednn_opt}"

# Key version numbers
ENV ACL_VERSION="v21.08" \
    OPENBLAS_VERSION=0.3.10 \
    NINJA_VERSION=1.9.0

# Package build parameters
ENV PROD_DIR=/opt \
    PACKAGE_DIR=packages

# Make directories to hold package source & build directories (PACKAGE_DIR)
# and install build directories (PROD_DIR).
RUN mkdir -p $PACKAGE_DIR && \
    mkdir -p $PROD_DIR

# Build Arm Optimized Routines from source
# provides optimised maths library fucntions for Aarch64
# see https://github.com/ARM-software/optimized-routines
COPY scripts/build-arm_opt_routines.sh $PACKAGE_DIR/.
COPY patches/config.mk $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-arm_opt_routines.sh

# Common compiler settings
ENV CC=gcc \
    CXX=g++ \
    BASE_CFLAGS="-mcpu=${CPU} -march=${ARCH} -O3" \    
    BASE_LDFLAGS="-L$PROD_DIR/arm_opt_routines/lib -lmathlib -lm" \
    LD_LIBRARY_PATH="$PROD_DIR/arm_opt_routines/lib"

# Build OpenBLAS from source
COPY scripts/build-openblas.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-openblas.sh
ENV OPENBLAS_DIR=$PROD_DIR/openblas
ENV LD_LIBRARY_PATH=$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH

# Build Arm Compute Library from source
COPY scripts/build-acl.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-acl.sh
ENV ACL_ROOT_DIR=$PROD_DIR/ComputeLibrary

# Build ninja from source
COPY scripts/build-ninja.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-ninja.sh
ENV PATH=$PROD_DIR/ninja/$NINJA_VERSION:$PATH


# ========
# Stage 3: install essential python dependencies into a venv
# ========
FROM deep-learning-libs AS deep-learning-tools
ARG njobs
ARG default_py_version
ARG cpu
ARG arch

ENV PY_VERSION="${default_py_version}" \
    NP_MAKE="${njobs}" \
    CPU="${cpu}" \
    ARCH="${arch}"

# Key version numbers
ENV NUMPY_VERSION=1.19.5 \
    SCIPY_VERSION=1.5.2 \
    NPY_DISTUTILS_APPEND_FLAGS=1 \
    OPENCV_VERSION=4.4.0.46

# Using venv means this can be done in userspace
WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER
ENV PACKAGE_DIR=/home/$DOCKER_USER/$PACKAGE_DIR
RUN mkdir -p $PACKAGE_DIR

# Setup a Python virtual environment
ENV VENV_DIR=/home/$DOCKER_USER/python3-venv
ENV VENV_ACT=$VENV_DIR/bin/activate
RUN python -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

# Install some basic python packages needed for NumPy
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "setuptools>=41.0.0" six mock wheel cython sh

# Build numpy from source, using OpenBLAS for BLAS calls
COPY scripts/build-numpy.sh $PACKAGE_DIR/.
COPY patches/site.cfg $PACKAGE_DIR/site.cfg
RUN $PACKAGE_DIR/build-numpy.sh

# Install some  basic python packages needed for SciPy
RUN pip install --no-cache-dir pybind11==2.6.2 pyangbind
# Build scipy from source, using OpenBLAS for BLAS calls
COPY scripts/build-scipy.sh $PACKAGE_DIR/.
COPY patches/site.cfg $PACKAGE_DIR/site.cfg
RUN $PACKAGE_DIR/build-scipy.sh

# Install some TensorFlow essentials
RUN pip install --no-cache-dir keras_applications==1.0.8 --no-deps
RUN pip install --no-cache-dir keras_preprocessing==1.1.2 --no-deps

# Install some more essentials.
RUN HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial pip install --no-cache-dir h5py==3.1.0
RUN pip install --no-cache-dir grpcio
RUN pip install --no-cache-dir hypothesis pyyaml pytest
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir pillow==8.2.0 lmdb
RUN pip install --no-cache-dir ck==1.55.5 absl-py pycocotools typing_extensions

RUN pip install --no-cache-dir scikit-build
RUN pip uninstall enum34 -y
RUN pip install --no-cache-dir opencv-python-headless==${OPENCV_VERSION}

CMD ["bash", "-l"]


# ========
# Stage 4: build TensorFlow and PyTorch
# ========
FROM deep-learning-libs AS deep-learning-dev
ARG njobs
ARG default_py_version
ARG cpu
ARG arch
ARG bazel_mem
ARG pt_onednn_opt
ARG tf_onednn_opt
ARG tf_version
ARG bazel_version
ARG eigen_l1_cache
ARG eigen_l2_cache
ARG eigen_l3_cache

ENV PT_ONEDNN_BUILD="${pt_onednn_opt}" \
    TF_ONEDNN_BUILD="${tf_onednn_opt}" \
    BZL_RAM="${bazel_mem}" \
    NP_MAKE="${njobs}" \
    CPU="${cpu}" \
    ARCH="${arch}" \
    EIGEN_L1_CACHE="${eigen_l1_cache}" \
    EIGEN_L2_CACHE="${eigen_l2_cache}" \
    EIGEN_L3_CACHE="${eigen_l3_cache}"

# Key version numbers
ENV PY_VERSION="${default_py_version}" \
    BZL_VERSION="${bazel_version}" \
    TF_VERSION="${tf_version}"

# Use a PACKAGE_DIR in userspace
WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER
ENV PACKAGE_DIR=/home/$DOCKER_USER/$PACKAGE_DIR
RUN mkdir -p $PACKAGE_DIR

# Copy in the Python virtual environment
ENV VENV_DIR=/home/$DOCKER_USER/python3-venv
COPY --chown=$DOCKER_USER:$DOCKER_USER --from=deep-learning-tools $VENV_DIR /home/$DOCKER_USER/python3-venv
ENV PATH="$VENV_DIR/bin:$PATH"

# Get Bazel binary for AArch64
COPY scripts/get-bazel.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/get-bazel.sh
ENV PATH=$PACKAGE_DIR/bazel:$PATH

# Build TensorFlow
COPY patches/eigen_workspace.patch $PACKAGE_DIR/.
COPY patches/eigen_gebp_cache.patch  $PACKAGE_DIR/.
COPY patches/tf_acl.patch $PACKAGE_DIR/.
COPY scripts/build-tensorflow.sh $PACKAGE_DIR/.
COPY patches/TF-caching.patch $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-tensorflow.sh

# Downgrade keras, tensorflow-estimator and tensorboard for tensorflow 2.6.0
RUN pip uninstall -y \
        keras \
        tensorflow-estimator \
        tensorboard && \
    pip install --no-cache-dir --no-deps \
        keras==2.6.0 \
        tensorflow-estimator==2.6.0
RUN pip install --no-cache-dir \
        tensorboard==2.6.0

# Build tensorflow-text
COPY scripts/build-tensorflow-text.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-tensorflow-text.sh

# Key version numbers
ENV TORCH_VERSION="1.9.0" \
    ONEDNN_VERSION="v2.4" \
    TORCH_VISION_VERSION="v0.9.1" \
    TORCH_TEXT_VERSION="0.10.0" \
    TORCH_AUDIO_VERSION="0.9.0"

# Build PyTorch
COPY scripts/build-pytorch.sh $PACKAGE_DIR/.
COPY patches/pocketfft_full.patch $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-pytorch.sh

# Build torchvision
RUN pip install --no-cache-dir git+https://github.com/pytorch/vision@$TORCH_VISION_VERSION

# Build torchtext
ENV VENV_PACKAGE_DIR=$VENV_DIR/lib/python$PY_VERSION/site-packages
COPY scripts/build-torchtext.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-torchtext.sh

# Build torchaudio
COPY scripts/build-torchaudio.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-torchaudio.sh

CMD ["bash", "-l"]


# ========
# Stage 5: Setup Coding Environment
# ========
FROM deep-learning-libs AS deep-learning-coding
ARG njobs

WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER

# Copy in the Python virtual environment
ENV VENV_DIR=/home/$DOCKER_USER/python3-venv
COPY --chown=$DOCKER_USER:$DOCKER_USER --from=deep-learning-dev $VENV_DIR /home/$DOCKER_USER/python3-venv
ENV PATH="$VENV_DIR/bin:$PATH"

# Install Rust into user-space, needed for transformers dependencies
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/$DOCKER_USER/.cargo/bin:${PATH}"

CMD ["bash", "-l"]


# ========
# Stage 6: Install benchmarks
# ========
FROM deep-learning-coding AS deep-learning-examples
ARG njobs

WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER
ENV PACKAGE_DIR=/home/$DOCKER_USER/package
ENV PROD_DIR=/opt
RUN mkdir -p $PACKAGE_DIR

# Build and install OpenCV from GitHub sources (needed for C++ API examples)
COPY scripts/build-opencv.sh $PACKAGE_DIR/.
RUN sudo -E $PACKAGE_DIR/build-opencv.sh
ENV LD_LIBRARY_PATH=$PROD_DIR/opencv/install/lib:$LD_LIBRARY_PATH

# Build and install yaml-cpp from Github source (needed for C++ API examples)
ENV YAML_VERSION=yaml-cpp-0.7.0
COPY scripts/build-yamlcpp.sh $PACKAGE_DIR/.
RUN sudo -E $PACKAGE_DIR/build-yamlcpp.sh

ENV LD_LIBRARY_PATH=$VENV_DIR/tensorflow/lib:$LD_LIBRARY_PATH

# Examples, benchmarks, and associated 'helper' scripts will be installed
# in $EXAMPLE_DIR.
ENV EXAMPLE_DIR=/home/$DOCKER_USER/examples
ENV MLCOMMONS_DIR=$EXAMPLE_DIR/MLCommons
RUN mkdir -p $EXAMPLE_DIR
RUN mkdir -p $MLCOMMONS_DIR
ADD examples $EXAMPLE_DIR

# Install missing Python package dependencies required to run examples
RUN pip install --no-cache-dir transformers pandas
RUN pip install --no-cache-dir pyyaml
RUN pip install --no-cache-dir requests
RUN pip install --no-cache-dir tqdm
RUN pip install --no-cache-dir boto3
RUN pip install --no-cache-dir future onnx==1.8.1
RUN pip install --no-cache-dir iopath

# Clone TensorFlow benchmarks into EXAMPLE_DIR.
COPY scripts/build-benchmarks.sh $EXAMPLE_DIR/.
RUN $EXAMPLE_DIR/build-benchmarks.sh
RUN rm -f $EXAMPLE_DIR/build-benchmarks.sh

# Clone and install MLCommons (MLPerf)
COPY patches/optional-mlcommons-changes.patch $MLCOMMONS_DIR/optional-mlcommons-changes.patch
COPY scripts/build-mlcommons.sh $MLCOMMONS_DIR/.
COPY patches/mlcommons_bert.patch $MLCOMMONS_DIR/.
COPY patches/pytorch_native.patch $MLCOMMONS_DIR/.
RUN $MLCOMMONS_DIR/build-mlcommons.sh
RUN rm -f $MLCOMMONS_DIR/build-mlcommons.sh

# Copy scripts to download dataset and models
COPY scripts/download-dataset.sh $MLCOMMONS_DIR/.
COPY scripts/download-model.sh $MLCOMMONS_DIR/.

# MLCommons ServerMode
COPY patches/Makefile.patch $MLCOMMONS_DIR/.
COPY patches/servermode.patch $MLCOMMONS_DIR/.
COPY scripts/build-boost.sh $MLCOMMONS_DIR/.
COPY scripts/build-loadgen-integration.sh $MLCOMMONS_DIR/.

# Copy scripts to download dataset and models
COPY scripts/download-dataset.sh $MLCOMMONS_DIR/.
COPY scripts/download-model.sh $MLCOMMONS_DIR/.
COPY scripts/setup-servermode.sh $MLCOMMONS_DIR/.

# Copy examples
COPY --chown=$DOCKER_USER:$DOCKER_USER examples $EXAMPLE_DIR

CMD ["bash", "-l"]
