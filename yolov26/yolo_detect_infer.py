from ultralytics import YOLO
import os
import torch
import cv2


if __name__ == "__main__":
    model_path = r"/home/work/ultralytics/runs/detect/runs/train/mobile/weights/best.pt"
    image_size = (640, 640)
    conf = 0.5
    iou = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)

    save_dir = r'results/pt'
    os.makedirs(save_dir, exist_ok=True)
    image_dir = r'ultralytics/assets'
    for image_file in os.listdir(image_dir):
        print(image_file)
        if not image_file.endswith('.jpg') and not image_file.endswith('.jepg') and not image_file.endswith('.png'):
            continue
        image_path = os.path.join(image_dir, image_file)
        # result = model.predict(image_path, save=True, imgsz=image_size, conf=conf, iou=iou, device=device)
        results = model(image_path)
        # out = results[0].plot()
        result = results[0]
        # print(result)
        boxes = result.boxes
        # print(f"boxes: {boxes}")
        scores = boxes.conf
        classes_ids = boxes.cls
        xyxys = boxes.xyxy
        for score, class_id, xyxy in zip(scores, classes_ids, xyxys):
            print(score, class_id, xyxy)

        
        # save_path = os.path.join(save_dir, image_file)
        # cv2.imwrite(save_path, out)
