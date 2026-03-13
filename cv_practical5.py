!pip install ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files
# Upload image
uploaded = files.upload()
# Get uploaded file name
image_path = list(uploaded.keys())[0]
# Load pre-trained YOLOv8 model
model = YOLO(&#39;yolov8n.pt&#39;)
# Read image
image = cv2.imread(image_path)
# Check if image loaded successfully
if image is None:
raise FileNotFoundError(&quot;Error: Image not loaded properly.&quot;)
# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Display original image
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title(&quot;Original Image&quot;)
plt.axis(&quot;off&quot;)
plt.show()
# Run YOLO detection
results = model(image_rgb)

# Get annotated image
annotated_image = results[0].plot()
# Display result
plt.figure(figsize=(10, 8))
plt.imshow(annotated_image)
plt.title(&quot;YOLO Object Detection&quot;)
plt.axis(&quot;off&quot;)
plt.show()
