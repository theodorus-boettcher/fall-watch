cmake_minimum_required(VERSION 3.22)
project(detector CXX)

# ncnn_DIR comes from local.properties (ncnn.dir) through app/build.gradle.
find_package(ncnn REQUIRED)

add_library(detector SHARED detector.cpp)
target_compile_features(detector PRIVATE cxx_std_17)
target_compile_options(detector PRIVATE -O2 -Wall -Wextra)
target_link_libraries(detector ncnn android log)
