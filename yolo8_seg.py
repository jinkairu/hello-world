from ultralytics import YOLO
import cv2
import numpy as np
# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model
out = model('002.jpg')
print(out[0].boxes.data[0])
print(out[0].boxes.cls)
xy = out[0].masks.xy
x1,y1,x2,y2 = out[0].boxes.data[0,:4]
names = out[0].names
cls = out[0].boxes.cls
img = cv2.imread("002.jpg")
xyxy = out[0].boxes.xyxy.tolist()
conf = out[0].boxes.conf.tolist()
pre_name = [names[int(i)] for i in cls]
for name_,conf_,xyxy_ in zip(pre_name,conf,xyxy):
    x1,y1,x2,y2 = xyxy_
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img, name_ + ': ' + '%.2f' % conf_, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2)

for i in xy[:]:
    for j in i[:]:
        cv2.circle(img,(int(j[0]),int(j[1])),1,(255,0,0),-1)
cv2.imshow('img',img)
cv2.imwrite('002_seg',img)
cv2.waitKey()
cv2.destroyAllWindows()


