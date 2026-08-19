from ultralytics import YOLO

# Load the downloaded model
model = YOLO("best.pt")

print("✅ Model Loaded Successfully!")
print("Classes detected by the model:")
print(model.names)