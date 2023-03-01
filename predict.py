import os

from ultralytics import YOLO
import cv2

# D:\runs\detect\train2\weights
model_path = os.path.join('D:/coin_detection_yolov8', 'runs', 'detect', 'train2', 'weights', 'best.pt')
print(model_path)
# Load a model
model = YOLO(model_path)  # load a custom model


img = 'D:/coin_detection_yolov8/runs/Bangladesh.jpg'

res = model(img)
res_plotted = res[0].plot()
cv2.imshow("result", res_plotted)
cv2.imwrite('D:/coin_detection_yolov8/output/coins.png',res_plotted)
