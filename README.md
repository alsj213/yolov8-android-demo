# YOLOv8 ONNX Runtime Android 实时目标检测

基于 ONNX Runtime 和 YOLOv8n 实现的 Android 实时目标检测应用。

## 特性

- 🚀 **实时检测**: 利用 CameraX 实现相机预览和实时检测
- ⚡ **高性能**: ONNX Runtime 优化，支持 CPU/NNAPI 加速
- 📊 **性能监控**: 实时 FPS 和推理时间显示
- 🔄 **模式切换**: 支持 CPU 和 NNAPI 硬件加速模式切换
- 🎯 **80类物体**: 支持 COCO 数据集 80 类物体检测

## 技术栈

- **模型**: YOLOv8n (640x640)
- **推理框架**: ONNX Runtime 1.19.2
- **相机**: CameraX 1.3.1
- **语言**: Kotlin
- **最低 SDK**: API 27 (Android 8.1)

## 项目结构

```
yolo/
├── app/                    # Android 应用
│   └── src/main/
│       ├── java/com/yolov8/demo/
│       │   ├── MainActivity.kt          # 主界面
│       │   ├── camera/
│       │   │   └── CameraManager.kt     # 相机管理
│       │   ├── detector/
│       │   │   ├── YOLOv8Detector.kt    # 推理封装
│       │   │   └── DetectorResult.kt    # 检测结果
│       │   ├── utils/
│       │   │   ├── ImageUtils.kt        # 图像工具
│       │   │   ├── NMSUtils.kt          # NMS后处理
│       │   │   └── FPSMonitor.kt        # 性能统计
│       │   └── view/
│       │       └── DetectionOverlayView.kt  # 检测框绘制
│       └── res/raw/yolov8n.onnx         # 模型文件
├── models/                # 模型转换脚本
└── docs/                  # 性能报告
```

## 构建和安装

### 前置要求
- Android SDK API 34+
- Android NDK (可选，用于原生优化)
- arm64-v8a 架构设备

### 构建步骤
```bash
# Windows
gradlew.bat assembleDebug

# Linux/Mac (WSL)
./gradlew assembleDebug
```

### 安装到设备
```bash
/mnt/e/andorid/adb/adb.exe install app/build/outputs/apk/debug/app-debug.apk
```

## 使用说明

1. 打开应用，授予相机权限
2. 相机自动启动，实时显示检测结果
3. 左上角显示 FPS 和推理时间
4. 点击右下角按钮切换 CPU/NNAPI 模式
5. 检测框显示物体类别和置信度

## 性能指标

预期性能（红米 K30s / Snapdragon 865）：

| 模式 | 平均 FPS | 平均推理时间 |
|------|----------|-------------|
| CPU (4线程) | ~15-20 | ~50-60ms |
| NNAPI | ~25-30 | ~30-40ms |

详细测试报告请参考 [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md)

## 优化建议

1. **模型优化**: 可以使用 YOLOv8n-int8 量化模型进一步提升速度
2. **预处理优化**: 使用 libyuv 替代 YuvImage 提升图像转换速度
3. **多线程**: 调整 numThreads 参数匹配 CPU 核心数
4. **NNAPI**: 在支持的设备上启用 NNAPI 获得显著性能提升

## 许可证

MIT License
