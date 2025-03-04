from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
out = model('output.jpeg')

names = out[0].names
cls = out[0].boxes.cls
'''
tensor([ 5.,  0.,  0.,  0., 11.,  0.])
'''
pre_name = [names[int(i)] for i in cls]
'''
['bus', 'person', 'person', 'person', 'stop sign', 'person']
'''
conf = out[0].boxes.conf.tolist()
xyxy = out[0].boxes.xyxy.tolist()
img = cv2.imread('output.jpeg')

for name_,conf_,xyxy_ in zip(pre_name,conf,xyxy):
    #if name_ != 'person': continue
    print(name_, conf_)
    x1,y1,x2,y2 = xyxy_
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img,name_+': '+'%.2f'%conf_,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

cv2.imshow('img', img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg',img)