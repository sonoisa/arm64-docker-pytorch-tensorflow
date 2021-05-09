# *******************************************************************************
# Copyright 2020 Arm Limited and affiliates.
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

FROM ubuntu:18.04 AS deep-learning-base

RUN if ! [ "$(arch)" = "aarch64" ] ; then exit 1; fi

#Install core OS packages
RUN apt-get -y update && \
    apt-get -y install software-properties-common --no-install-recommends && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get -y install --no-install-recommends \
      autoconf \
      bc \
      build-essential \
      cmake \
      curl \
      g++-9 \
      gcc-9 \
      gettext-base \
      gfortran-9 \
      git \
      iputils-ping \
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
      libreadline-dev \
      libsox-dev \
      libsox-fmt-all \
      libssl-dev \
      libsqlite3-dev \
      libxml2-dev \
      libxslt-dev \
      locales \
      moreutils \
      openjdk-8-jdk \
      openssl \
      python-openssl \
      rsync \
      scons \
      sox \
      ssh \
      sudo \
      time \
      unzip \
      vim \
      wget \
      xz-utils \
      zip \
      zlib1g-dev \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# Make gcc 9 the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 1

# DOCKER_USER for the Docker user
ENV DOCKER_USER=ubuntu

# Setup default user
RUN useradd --create-home -s /bin/bash -m $DOCKER_USER && echo "$DOCKER_USER:Arm2020" | chpasswd && adduser $DOCKER_USER sudo
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
ARG pt_onednn_opt
ARG tf_onednn_opt
ARG onednn_version
ARG cpu
ARG tf_id

ENV NP_MAKE="${njobs}" \
    PT_ONEDNN_BUILD="${pt_onednn_opt}" \
    TF_ONEDNN_BUILD="${tf_onednn_opt}" \
    ONEDNN_VERSION="${onednn_version}" \
    CPU="${cpu}" \
    TF_VERSION_ID="${tf_id}"

# Key version numbers
ENV PY_VERSION=3.8.7 \
    ACL_VERSION="v20.08" \
    ARMPL_VERSION=20.2.1 \
    OPENBLAS_VERSION=0.3.9 \
    NINJA_VERSION=1.9.0

# Package build parameters
ENV PROD_DIR=/opt \
    PACKAGE_DIR=/packages

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
    BASE_CFLAGS="-mcpu=${CPU} -moutline-atomics"

# Install Arm Performance Libraries
COPY scripts/build-armpl.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-armpl.sh
ENV ARMPL_DIR=$PROD_DIR/armpl/armpl_${ARMPL_VERSION}_gcc-9.3

# Common compiler settings for remaining builds
# this ads arm_opt_routined into the LDFLAGS by default.
ENV BASE_LDFLAGS="-L$ARMPL_DIR/lib -L$PROD_DIR/arm_opt_routines/lib -lmathlib -lm" \
    LD_LIBRARY_PATH="$ARMPL_DIR/lib:$PROD_DIR/arm_opt_routines/lib"

COPY scripts/build-cpython.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-cpython.sh

# Build OpenBLAS from source
COPY scripts/build-openblas.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-openblas.sh
ENV OPENBLAS_DIR=$PROD_DIR/openblas/$OPENBLAS_VERSION
ENV LD_LIBRARY_PATH=$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH

# Build Arm Compute Library from source
COPY scripts/build-acl.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-acl.sh
ENV ACL_ROOT_DIR=$PROD_DIR/ComputeLibrary

# Build ninja from source
COPY scripts/build-ninja.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-ninja.sh
ENV PATH=$PROD_DIR/ninja/$NINJA_VERSION:$PATH

# Build oneDNN from source
COPY scripts/build-onednn.sh $PACKAGE_DIR/.
# Patch for oneDNN (MKL-DNN), v0.21.3
COPY patches/mkldnn.patch $PACKAGE_DIR/mkldnn.patch
ENV ONEDNN_DIR=$PROD_DIR/onednn/release
ENV LD_LIBRARY_PATH=$ONEDNN_DIR/lib:$LD_LIBRARY_PATH
# Get oneDNN sources
# Built as part of the bazel build step in TensorFlow
COPY scripts/get-onednn.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/get-onednn.sh

# ========
# Stage 3: install essential python dependencies into a venv
# ========
FROM deep-learning-libs AS deep-learning-tools
ARG njobs
ENV NP_MAKE="${njobs}"

# Key version numbers
ENV NUMPY_VERSION=1.18.5 \
    SCIPY_VERSION=1.4.1 \
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
RUN pip install --no-cache-dir "setuptools>=41.0.0" six mock wheel cython

# Build numpy from source, using OpenBLAS for BLAS calls
COPY scripts/build-numpy.sh $PACKAGE_DIR/.
COPY patches/site.cfg $PACKAGE_DIR/site.cfg
RUN $PACKAGE_DIR/build-numpy.sh

# Install some  basic python packages needed for SciPy
RUN pip install --no-cache-dir pybind11 pyangbind
# Build numpy from source, using OpenBLAS for BLAS calls
COPY scripts/build-scipy.sh $PACKAGE_DIR/.
COPY patches/site.cfg $PACKAGE_DIR/site.cfg
RUN $PACKAGE_DIR/build-scipy.sh

# Install some TensorFlow essentials
RUN pip install --no-cache-dir keras_applications==1.0.8 --no-deps
RUN pip install --no-cache-dir keras_preprocessing==1.1.0 --no-deps

# Install some more essentials.
RUN HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial pip install --no-cache-dir h5py==2.10.0
RUN pip install --no-cache-dir grpcio
RUN pip install --no-cache-dir hypothesis pyyaml pytest
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir lmdb pillow==6.1
RUN pip install --no-cache-dir ck absl-py pycocotools 

# Install OpenCV into our venv,
# Note: Scripts are provided to build and install OpenCV from the
# GitHub repository. These are no longer used by default in favour of the
# opencv-python package, in this case opencv-python-headless.
# Uncomment code block '1' below, and comment out code block '2' to build
# from the GitHub sources.
# --
# 1 - build from GitHub sources:
#COPY scripts/build-opencv.sh $PACKAGE_DIR/.
#RUN $PACKAGE_DIR/build-opencv.sh
# --
# 2 - install opencv-python-headless
RUN pip install --no-cache-dir scikit-build
# enum34 is not compatable with Python 3.6+, and not required
# it is installed as a dependency for an earlier package and needs
# to be removed in order for the OpenCV build to complete.
RUN pip uninstall enum34 -y
RUN pip install --no-cache-dir --no-binary :all: opencv-python-headless==${OPENCV_VERSION}

CMD ["bash", "-l"]

# ========
# Stage 4: build TensorFlow and PyTorch
# ========
FROM deep-learning-libs AS deep-learning-dev
ARG njobs
ARG bazel_mem
ARG pt_onednn_opt
ARG tf_onednn_opt
ARG tf_version
ARG tf_id
ARG bazel_version


ENV PT_ONEDNN_BUILD="${pt_onednn_opt}" \
    TF_ONEDNN_BUILD="${tf_onednn_opt}" \
    BZL_RAM="${bazel_mem}" \
    NP_MAKE="${njobs}"

# Key version numbers
ENV BZL_VERSION="${bazel_version}" \
    TF_VERSION="${tf_version}" \
    TF_VERSION_ID="${tf_id}"

# Key version numbers
ENV TORCH_VERSION=1.6.0
ENV TORCH_VISION_VERSION=v0.7.0
ENV TORCH_TEXT_VERSION=v0.7.0-rc3
ENV TORCH_AUDIO_VERSION=v0.6.0

# Use a PACKAGE_DIR in userspace
WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER
ENV PACKAGE_DIR=/home/$DOCKER_USER/$PACKAGE_DIR
RUN mkdir -p $PACKAGE_DIR

# Copy in the Python virtual environment
ENV VENV_DIR=/home/$DOCKER_USER/python3-venv
COPY --chown=$DOCKER_USER:$DOCKER_USER --from=deep-learning-tools $VENV_DIR /home/$DOCKER_USER/python3-venv
ENV PATH="$VENV_DIR/bin:$PATH"

# Build PyTorch
COPY scripts/build-pytorch.sh $PACKAGE_DIR/.
COPY patches/pytorch_onednn.patch $PACKAGE_DIR/.
COPY patches/onednn_acl_verbose.patch $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-pytorch.sh

RUN pip install --no-cache-dir git+https://github.com/pytorch/text@$TORCH_TEXT_VERSION
RUN pip install --no-cache-dir git+https://github.com/pytorch/vision@$TORCH_VISION_VERSION
RUN pip install --no-cache-dir git+https://github.com/pytorch/audio@$TORCH_AUDIO_VERSION

# Build Bazel
COPY scripts/build-bazel.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-bazel.sh
ENV PATH=$PACKAGE_DIR/bazel/output:$PATH

# Build TensorFlow
COPY scripts/build-tensorflow.sh $PACKAGE_DIR/.
COPY patches/tf_dnnl_decoupling.patch $PACKAGE_DIR/tf_dnnl_decoupling.patch
COPY patches/tf2_onednn_decoupling.patch $PACKAGE_DIR/tf2_onednn_decoupling.patch
COPY patches/tensorflow.patch $PACKAGE_DIR/tensorflow.patch
COPY patches/tensorflow2.patch $PACKAGE_DIR/tensorflow2.patch
# Patches to resolve intel binary blob dependencies for TensorFlow 2.x - oneDNN 1.x builds
COPY patches/oneDNN-opensource.patch $PACKAGE_DIR/oneDNN-opensource.patch
# Patches to support different flavours of oneDNN
COPY patches/tf2-armpl.patch $PACKAGE_DIR/tf2-armpl.patch
COPY patches/tf2-openblas.patch $PACKAGE_DIR/tf2-openblas.patch
RUN $PACKAGE_DIR/build-tensorflow.sh

# Build tensorflow-text
COPY scripts/build-tensorflow-text.sh $PACKAGE_DIR/.
RUN $PACKAGE_DIR/build-tensorflow-text.sh

RUN rm -rf $PACKAGE_DIR/bazel

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

CMD ["bash", "-l"]


# ========
# Stage 6: Install benchmarks
# ========
FROM deep-learning-coding AS deep-learning-examples
ARG njobs

WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER

# Clone Tensorflow benchmarks
COPY scripts/build-benchmarks.sh /home/$DOCKER_USER/.
RUN /home/$DOCKER_USER/build-benchmarks.sh

COPY patches/optional-mlcommons-changes.patch /home/$DOCKER_USER/optional-mlcommons-changes.patch
# Clone and install  MLCommons (MLPerf)
COPY scripts/build-mlcommons.sh /home/$DOCKER_USER/.
RUN /home/$DOCKER_USER/build-mlcommons.sh

# Copy scripts to download dataset and models
COPY scripts/download-dataset.sh /home/$DOCKER_USER/.
COPY scripts/download-model.sh /home/$DOCKER_USER/.

# Clone PyTorch examples
# NOTE: these examples are included as a startiing point
RUN  git clone https://github.com/pytorch/examples.git

CMD ["bash", "-l"]
