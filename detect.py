#!/usr/bin/python
import cv2
from picamera2 import Picamera2
import time
from ultralytics import YOLO

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
picam2.framerate = 30
# Load the YOLOv8 model
model = YOLO("yolov5nu.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
xncnn_model = YOLO("yolov5nu.torchscript")

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()

    # Run YOLOv8 inference on the frame
    results = xncnn_model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)
    #time.sleep(0.1)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()
