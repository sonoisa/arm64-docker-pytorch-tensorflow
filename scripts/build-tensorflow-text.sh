#!/usr/bin/env bash

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

set -euo pipefail

cd $PACKAGE_DIR
readonly package=tensorflow-text
readonly version=$TF_VERSION
readonly tf_id=$TF_VERSION_ID
readonly src_host=https://github.com/tensorflow
readonly src_repo=text

# Clone tensorflow-text
git clone ${src_host}/${src_repo}.git
cd ${src_repo}
git checkout $version -b $version


# Env vars used to avoid interactive elements of the build.
export HOST_C_COMPILER=(which gcc)
export HOST_CXX_COMPILER=(which g++)
export PYTHON_BIN_PATH=(which python)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export CC_OPT_FLAGS=""
export TF_ENABLE_XLA=0
export TF_NEED_GCP=0
export TF_NEED_S3=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_CUDA=0
export TF_DOWNLOAD_CLANG=0
export TF_NEED_MPI=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_ROCM=0

./oss_scripts/configure.sh

extra_args=""
host_args=""
if [[ $BZL_RAM ]]; then extra_args="$extra_args --local_ram_resources=$BZL_RAM"; fi
if [[ $BZL_RAM ]]; then host_args="--host_jvm_args=-Xmx${BZL_RAM}m --host_jvm_args=-Xms${BZL_RAM}m"; fi
if [[ $NP_MAKE ]]; then extra_args="$extra_args --jobs=$NP_MAKE"; fi

if [[ $TF_ONEDNN_BUILD ]]; then
   echo "$TF_ONEDNN_BUILD build for $TF_VERSION"
   if [[ $tf_id == '1' ]]; then
      bazel $host_args build $extra_args \
         --verbose_failures \
         --define=build_with_mkl_dnn_only=true --define=build_with_mkl=true \
         --define=tensorflow_mkldnn_contraction_kernel=1 \
         --copt="-mtune=${CPU}" --copt="-march=armv8-a" --copt="-moutline-atomics" \
         --cxxopt="-mtune=${CPU}" --cxxopt="-march=armv8-a" --cxxopt="-moutline-atomics" \
         --linkopt="-L$ARMPL_DIR/lib -lamath -lm" --linkopt="-fopenmp" \
         --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
         --enable_runfiles oss_scripts/pip_package:build_pip_package
   elif [[ $tf_id == '2' ]]; then
      bazel $host_args build $extra_args \
         --verbose_failures \
         --config=mkl_opensource_only \
         --copt="-mcpu=${CPU}" --copt="-flax-vector-conversions" --copt="-moutline-atomics" --copt="-O3" \
         --cxxopt="-mcpu=${CPU}" --cxxopt="-flax-vector-conversions" --cxxopt="-moutline-atomics" --cxxopt="-O3" \
         --linkopt="-L$ARMPL_DIR/lib -lamath -lm" --linkopt="-fopenmp" \
         --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
         --enable_runfiles oss_scripts/pip_package:build_pip_package
   else
      echo 'Invalid TensorFlow version when building tensorflow'
      exit 1
   fi
else
   echo "Eigen-only build for $TF_VERSION"
   bazel $host_args build $extra_args \
      --verbose_failures \
      --define tensorflow_mkldnn_contraction_kernel=0 \
      --copt="-mcpu=${CPU}" --copt="-flax-vector-conversions" --copt="-moutline-atomics" --copt="-O3" \
      --cxxopt="-mcpu=${CPU}" --cxxopt="-flax-vector-conversions" --cxxopt="-moutline-atomics" --cxxopt="-O3" \
      --linkopt="-L$ARMPL_DIR/lib -lamath -lm" \
      --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
      --enable_runfiles oss_scripts/pip_package:build_pip_package
fi
./bazel-bin/oss_scripts/pip_package/build_pip_package ./wheel-TFT$TF_VERSION-py$PY_VERSION-$CC

pip install $(ls -tr wheel-TFT$TF_VERSION-py$PY_VERSION-$CC/*.whl | tail)

# Check the installation was sucessfull
cd $HOME

# if python -c 'import tensorflow_text; print(tensorflow_text.__version__)' > version.log; then
#    echo "TensorFlow-text $(cat version.log) package installed from $TF_VERSION branch."
# else
#    echo "TensorFlow-text package installation failed."
#    exit 1
# fi
#
# rm $HOME/version.log

rm -rf $PACKAGE_DIR/${src_repo}
