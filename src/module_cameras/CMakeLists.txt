cmake_minimum_required(VERSION 3.27)

project(module_cameras)

# 定义项目版本号宏
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

# 生成版本号字符串
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Project Dir: ${CMAKE_CURRENT_SOURCE_DIR}")

# obsensor sdk
set(OrbbecSDK_ROOT_DIR ${CMAKE_SOURCE_DIR}/3rdparty/orbbec)
set(OrbbecSDK_INCLUDE_DIR ${OrbbecSDK_ROOT_DIR}/include)
set(OrbbecSDK_LIBRARY_DIRS ${OrbbecSDK_ROOT_DIR}/lib_x64)
file(GLOB ORBBEC_LIBS ${OrbbecSDK_LIBRARY_DIRS}/lib*.so)
message(STATUS "library: ${ORBBEC_LIBS}")

# 将源代码添加到此项目的可执行文件。
file(GLOB FILES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB FILES_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

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
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${OrbbecSDK_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 链接库文件夹
# 如果动态链接库和静态链接库同名，优先调用静态链接库
target_link_directories(${PROJECT_NAME}
        PRIVATE
        ${OrbbecSDK_LIBRARY_DIRS})

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        pthread
        ${ORBBEC_LIBS}
        ${OpenCV_LIBS})

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES ${FILES_HEADERS}
        DESTINATION include/${PROJECT_NAME})
