cmake_minimum_required(VERSION 3.22.1)
project(yolovulkan)

set(CMAKE_CXX_STANDARD 17)

set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolovulkan SHARED
    yolo.cpp
    yolo_jni.cpp
)

find_library(log-lib log)

target_link_libraries(yolovulkan
    ncnn
    jnigraphics
    android
    ${log-lib}
)
