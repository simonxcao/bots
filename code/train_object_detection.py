# Use roboflow to tag data and then download it in yolov11n format, then you can run the object detector
from ultralytics import YOLO

def main():
	model = YOLO("yolo11n.pt")  # Load a YOLOv11 pre-trained model or your custom model
	# model.train(data="datasets/cuphead/data.yaml", epochs=50, imgsz=640, rect=True)
	model.train(data="datasets/cuphead/data.yaml", imgsz=(1587, 918), epochs=400, rect=True)

if __name__ == '__main__':
	main()