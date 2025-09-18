#!/bin/bash

echo "Setting up FlashOCC environment with C++11 and ROS2 support..."

# 激活conda环境
conda activate FlashOcc-py310

# 设置C++编译器 (使用系统GCC以避免CUDA编译问题)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# 加载ROS2环境
source /opt/ros/humble/setup.bash

# 设置CUDA环境
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr
export CUDA_HOME=/usr
export CUDA_INCLUDE_DIRS=/usr/include
export CUDA_PATH=/usr

echo "Environment setup complete!"

# 验证环境
echo "Python version: $(python --version)"
echo "GCC version: $($CXX --version | head -1)"
echo "ROS2 available: $(python -c 'import rclpy; print("Yes")' 2>/dev/null || echo "No")"
