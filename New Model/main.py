import torch
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from twilio.rest import Client

model1 = YOLO("yolov8n.pt")
mobilenet_model = load_model("crash_detection_mobilenetv2.h5")

video_path = "./sample.mp4"
cap = cv2.VideoCapture(video_path)

# Parameters for distance estimation
real_world_object_width = 1.5 # in meters
focal_length = 800  # in pixels

# Function to estimate distance from the camera
def estimate_distance(object_width, object_bbox_width):
    distance = (real_world_object_width * focal_length) / object_bbox_width
    return distance

# Vehicle class IDs in COCO dataset
vehicle_class_ids = [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

def sendAlert(msg):
    print(msg)
    account_sid = 'ACab0415f8ffa2085e6fbb50dd880d4e36'
    auth_token = 'a63ad1af67928a50199f0a795a46ee46'
    client = Client(account_sid, auth_token)
    message = client.messages.create(
    from_='+17088157804',
    body=msg,
    to='+919600071484'
    )

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    results = model1(frame)

    # Define danger zone (bottom-center)
    danger_zone_top = int(height * 0.75)  
    overlay = frame.copy()
    collision_occured = False
    cv2.rectangle(overlay, (0, danger_zone_top), (width, height), (0, 0, 255), thickness=-1)
    collision_frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    for result in results:
        for box in result.boxes:
            x, y, w, h = map(int, box.xywh[0])
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = model1.names[class_id]

            if class_id not in vehicle_class_ids:
                continue

            color = (0, 255, 0)
            
            obj_crop = frame[y1:y2, x1:x2]
            if obj_crop.size > 0:
                obj_crop = cv2.resize(obj_crop, (224, 224))
                img_array = image.img_to_array(obj_crop)
                img_array = np.expand_dims(img_array, axis=0) / 255.0  
                prediction = mobilenet_model.predict(img_array)[0][0]
    
                # Estimate distance
                object_distance = estimate_distance(real_world_object_width, w)

                print(object_distance, prediction)
                print(collision_occured)
                # Check danger zone and other conditions
                if object_distance < 3.5:
                    text = f"Collision !!"
                    if not collision_occured:
                        sendAlert("Alert!! Collision Detected at OMR Chennai. Impact of Collision : Severe")
                        collision_occured = True
                    cv2.putText(collision_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif y2 > danger_zone_top and prediction > 0.75 and object_distance <= 3.7:
                    color = (0, 0, 255)
                    text = f"Collision Risk !!"
                    cv2.putText(collision_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.rectangle(collision_frame, (x1, y1), (x2, y2), color, 3)
                elif object_distance < 10.5:
                    distance_text = f"Distance: {object_distance:.2f}m"
                    cv2.putText(collision_frame, distance_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Collision Detection with Distance", collision_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  

cap.release()
cv2.destroyAllWindows()