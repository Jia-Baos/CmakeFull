cmake_minimum_required(VERSION 3.27)

# 进入三方库文件夹
add_subdirectory(module_utils)
add_subdirectory(module_cameras)
add_subdirectory(module_inference)

# 进入执行文件文件夹
add_subdirectory(demo)
