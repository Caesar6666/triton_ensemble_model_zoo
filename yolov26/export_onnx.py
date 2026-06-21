from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolo26s.pt')
    model.export(
        format='onnx',
        imgsz=(640,640),
        dynamic=True,
        half=True,
        nms=True,
        opset=12
        )