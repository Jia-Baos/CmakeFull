cmake_minimum_required(VERSION 3.27)

project(test_module_inference_nanodet_ncnn)

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project Dir: ${CMAKE_CURRENT_SOURCE_DIR}")

# 将源代码添加到此项目的可执行文件。
file(GLOB FILES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${FILES_SRC})

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
        message(STATUS "OpenCV library status: ")
        message(STATUS " version: ${OpenCV_VERSION}")
        # message(STATUS " library: ${OpenCV_LIBS}")
        message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")
else()
        message(STATUS "Could not find OpenCV library")
endif()

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/3rdparty/NCNN/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
        message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
        message("ncnn FOUND ")
endif()

# 链接库头文件路径
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/module_inference/include
        ${CMAKE_SOURCE_DIR}/src/module_inference/include/det
        ${CMAKE_SOURCE_DIR}/src/module_inference/include/det/ncnn
        ${CMAKE_INSTALL_PREFIX}/include/module_inference
        ${CMAKE_INSTALL_PREFIX}/include/module_inference/det
        ${CMAKE_INSTALL_PREFIX}/include/module_inference/det/ncnn)

# 链接库文件夹
# 如果动态链接库和静态链接库同名，优先调用静态链接库
target_link_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_INSTALL_PREFIX}/lib)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ncnn
        ${OpenCV_LIBS}
        module_inference)

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
