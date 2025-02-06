# Use roboflow to tag data and then download it in yolov11n format, then you can run the object detector
from ultralytics import YOLO

def main():
	model = YOLO("yolo11n.pt")  # Load a YOLOv11 pre-trained model or your custom model
	# imgsz maintains the aspect ratio of the image. The images saved had this width
	model.train(data="datasets/cuphead/data.yaml", imgsz=1587, epochs=400, rect=True)

if __name__ == '__main__':
	main()