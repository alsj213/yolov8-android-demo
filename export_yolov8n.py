#!/usr/bin/env python3
"""
Export YOLOv8n model to ONNX format for ONNX Runtime
"""
import sys
import os

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system(f"{sys.executable} -m pip install ultralytics")
    from ultralytics import YOLO

def main():
    # Load YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Export to ONNX
    # Note: Using simplify=True for better ORT performance
    output_path = model.export(
        format='onnx',
        imgsz=640,
        batch=1,
        simplify=True,
        opset=17,
        dynamic=False
    )

    print(f"\nModel exported to: {output_path}")

    # Move to app resources
    app_raw_dir = "../app/src/main/res/raw/"
    if os.path.exists(app_raw_dir):
        dest_path = os.path.join(app_raw_dir, "yolov8n.onnx")
        import shutil
        shutil.copy(output_path, dest_path)
        print(f"Copied to: {dest_path}")

    # Print COCO class names
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    with open('coco_classes.txt', 'w') as f:
        f.write('\n'.join(coco_classes))
    print("\nCOCO class names saved to: coco_classes.txt")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
