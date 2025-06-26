#!/usr/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/3rdparty/orbbec/lib_x64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/3rdparty/MNN/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/3rdparty/NCNN/lib/cmake/ncnn

./install/bin/test_module_cameras
