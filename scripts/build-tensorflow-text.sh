#!/usr/bin/env bash

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

set -euo pipefail

cd $PACKAGE_DIR
readonly package=tensorflow-text
readonly version=$TF_VERSION
readonly src_host=https://github.com/tensorflow
readonly src_repo=text

# Clone tensorflow and benchmarks
git clone ${src_host}/${src_repo}.git
cd ${src_repo}
git checkout $version -b $version
git submodule sync
git submodule update --init --recursive

# Env vars used to avoid interactive elements of the build.
export HOST_C_COMPILER=(which gcc)
export HOST_CXX_COMPILER=(which g++)
export PYTHON_BIN_PATH=(which python)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_ENABLE_XLA=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_GCP=0
export TF_NEED_S3=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_CUDA=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_NEED_JEMALLOC=1
export TF_NEED_VERBS=0
export TF_NEED_AWS=0
export TF_NEED_GDR=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_KAFKA=0
export TF_NEED_TENSORRT=0

./oss_scripts/configure.sh

host_args=""
extra_args="--verbose_failures -s"
if [[ $BZL_RAM ]]; then extra_args="$extra_args --local_ram_resources=$BZL_RAM"; fi
if [[ $BZL_RAM ]]; then host_args="--host_jvm_args=-Xmx${BZL_RAM}m --host_jvm_args=-Xms${BZL_RAM}m"; fi
if [[ $NP_MAKE ]]; then extra_args="$extra_args --jobs=$NP_MAKE"; fi

if [[ $TF_ONEDNN_BUILD ]]; then
    echo "$TF_ONEDNN_BUILD build for $TF_VERSION"
    # extra_args="$extra_args --config=mkl_aarch64 --linkopt=-fopenmp"
    extra_args="$extra_args --define=build_with_mkl_dnn_only=true --define=build_with_mkl=true \
         --define=tensorflow_mkldnn_contraction_kernel=1 --linkopt=-fopenmp"
    if [[ $TF_ONEDNN_BUILD == 'reference' ]]; then
      echo "TensorFlow-text $TF_VERSION with oneDNN backend - reference build."
    elif [[ $TF_ONEDNN_BUILD == 'acl' ]]; then
      echo "TensorFlow-text $TF_VERSION with oneDNN backend - Compute Library build."
    fi
else
    echo "TensorFlow $TF_VERSION with Eigen backend."
    extra_args="$extra_args --define tensorflow_mkldnn_contraction_kernel=0"

    # Manually set L1,2,3 caches sizes for the GEBP kernel in Eigen.
    [[ $EIGEN_L1_CACHE ]] && extra_args="$extra_args \
      --cxxopt=-DEIGEN_DEFAULT_L1_CACHE_SIZE=${EIGEN_L1_CACHE} \
      --copt=-DEIGEN_DEFAULT_L1_CACHE_SIZE=${EIGEN_L1_CACHE}"
    [[ $EIGEN_L2_CACHE ]] && extra_args="$extra_args \
      --cxxopt=-DEIGEN_DEFAULT_L2_CACHE_SIZE=${EIGEN_L2_CACHE} \
      --copt=-DEIGEN_DEFAULT_L2_CACHE_SIZE=${EIGEN_L2_CACHE}"
    [[ $EIGEN_L3_CACHE ]] && extra_args="$extra_args \
      --cxxopt=-DEIGEN_DEFAULT_L3_CACHE_SIZE=${EIGEN_L3_CACHE} \
      --copt=-DEIGEN_DEFAULT_L3_CACHE_SIZE=${EIGEN_L3_CACHE}"
fi

# Build the tensorflow configuration
bazel $host_args build $extra_args \
        --copt="-mcpu=${CPU}" --copt="-march=${ARCH}" --copt="-O3"  --copt="-fopenmp" \
        --cxxopt="-mcpu=${CPU}" --cxxopt="-march=${ARCH}" --cxxopt="-O3"  --cxxopt="-fopenmp" \
        --linkopt="-lgomp  -lm" \
        --enable_runfiles \
        oss_scripts/pip_package:build_pip_package

# Install Tensorflow-text python package via pip
./bazel-bin/oss_scripts/pip_package/build_pip_package ./wheel-TFT$TF_VERSION-py$PY_VERSION-$CC
pip install $(ls -tr wheel-TFT$TF_VERSION-py$PY_VERSION-$CC/*.whl | tail)

# Check the Python installation was sucessfull
cd $HOME
if python -c 'import tensorflow_text; print(tensorflow_text.__version__)' > version.log; then
    echo "TensorFlow-text $(cat version.log) package installed from $TF_VERSION branch."
else
    echo "TensorFlow-text Python package installation failed."
    exit 1
fi
rm $HOME/version.log

rm -rf $PACKAGE_DIR/${src_repo}
