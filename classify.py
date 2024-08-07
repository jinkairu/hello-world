from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-cls.pt")  # load an official model
results = model("output.jpeg")
print(results[0].probs.data.sum())
