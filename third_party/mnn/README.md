# MNN (Mobile Neural Network) Integration

## Source
- MNN version: 2.5.1
- Source: https://github.com/alibaba/MNN

## Files
- `include/` - MNN header files
- `mnnnetnative.cpp` - JNI bindings for MNN Java API
- `jniLibs/arm64-v8a/libmnncore.so` - Compiled JNI bindings (arm64-v8a)

## Building JNI Bindings

```bash
# Set NDK path
export NDK=/path/to/android-ndk

# Compile with NDK clang
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android27-clang++ \
  -shared -fPIC -std=c++17 \
  -Iinclude \
  -lMNN -llog -ljnigraphics \
  -o jniLibs/arm64-v8a/libmnncore.so \
  mnnnetnative.cpp
```

## libMNN.so Compilation

```bash
cd MNN
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-27 \
  -DMNN_BUILD_SHARED_LIBS=ON \
  ..
make -j8
```
