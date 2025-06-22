cmake_minimum_required(VERSION 3.27)

project(module_utils)

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project Dir: ${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB FILES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/include/*.cpp)

file(GLOB FILES_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

install(FILES ${FILES_HEADERS}
        DESTINATION include/${PROJECT_NAME})
