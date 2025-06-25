cmake_minimum_required(VERSION 3.27)

project(module_inference)

option(ENABLE_MNN "support mnn model" ON)
option(ENABLE_NCNN "support ncnn model" ON)

# 定义项目版本号宏
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

# 生成版本号字符串
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Project Dir: ${CMAKE_CURRENT_SOURCE_DIR}")

# 将源代码添加到此项目的可执行文件。
file(GLOB FILES_SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/det/*.cpp)
file(GLOB FILES_HEADERS_BASE ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

if(ENABLE_MNN)
    message(STATUS "MNN enabled")
    add_compile_definitions(USE_MNN)
    file(GLOB FILES_SRC_DET_MNN ${CMAKE_CURRENT_SOURCE_DIR}/src/det/mnn/*.cpp)
    file(GLOB FILES_HEADERS_DET_MNN ${CMAKE_CURRENT_SOURCE_DIR}/include/det/mnn/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/det/mnn/*.hpp)

    set(MNN_DIR ${CMAKE_SOURCE_DIR}/3rdparty/MNN)
endif()

if (ENABLE_NCNN)
    message(STATUS "NCNN enabled")
    add_compile_definitions(USE_NCNN)
    file(GLOB FILES_SRC_DET_NCNN ${CMAKE_CURRENT_SOURCE_DIR}/src/det/ncnn/*.cpp)
    file(GLOB FILES_HEADERS_DET_NCNN ${CMAKE_CURRENT_SOURCE_DIR}/include/det/ncnn/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/det/ncnn/*.hpp)

    set(ncnn_DIR ${CMAKE_SOURCE_DIR}/3rdparty/NCNN/lib/cmake/ncnn)
    
    find_package(ncnn REQUIRED)
        if(NOT TARGET ncnn)
        message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
        else()
        message("ncnn FOUND ")
        endif()
endif()

set(FILES_SRC ${FILES_SRC_BASE} ${FILES_SRC_DET_MNN} ${FILES_SRC_DET_NCNN})

add_library(${PROJECT_NAME} SHARED ${FILES_SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}.${PROJECT_VERSION}")

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
        message(STATUS "OpenCV library status: ")
        message(STATUS " version: ${OpenCV_VERSION}")
        # message(STATUS " library: ${OpenCV_LIBS}")
        message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")
else()
        message(STATUS "Could not find OpenCV library")
endif()

# 链接库头文件路径
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/det
        ${CMAKE_CURRENT_SOURCE_DIR}/include/det/mnn
        ${MNN_DIR}/include/MNN
        )

# 链接库文件夹，如果动态链接库和静态链接库同名，优先调用静态链接库
target_link_directories(${PROJECT_NAME}
        PRIVATE
        ${MNN_DIR}/lib
        )

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${OpenCV_LIBS}
        MNN
        ncnn)

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES ${FILES_HEADERS_BASE}
        DESTINATION include/${PROJECT_NAME})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/det/model_det_zoo.hpp
        DESTINATION include/${PROJECT_NAME}/det)

install(FILES ${FILES_HEADERS_DET_MNN}
        DESTINATION include/${PROJECT_NAME}/det/mnn)

install(FILES ${FILES_HEADERS_DET_NCNN}
        DESTINATION include/${PROJECT_NAME}/det/ncnn)
