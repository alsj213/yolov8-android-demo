cmake_minimum_required(VERSION 3.4.1)
project(mnncore)

# MNN headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# JNI source
add_library(
    mnncore
    SHARED
    src/mnnnetnative.cpp
)

# Link against prebuilt MNN
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../app/src/main/jniLibs/${ANDROID_ABI}/libMNN.so)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(mnncore MNN ${log-lib} ${jnigraphics-lib})
