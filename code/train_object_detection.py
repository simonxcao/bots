# Use roboflow to tag data and then download it in yolov8n format, then you can run the object detector
import torch
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # Load a YOLOv8 pre-trained model or your custom model
    model.train(data="datasets/cuphead/data.yaml", epochs=50, imgsz=640, rect=True)

if __name__ == '__main__':
    main()
