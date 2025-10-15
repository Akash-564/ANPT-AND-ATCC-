# from ultralytics import YOLO
# import cv2

# import util
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv


# results = {}

# mot_tracker = Sort()

# # load models
# coco_model = YOLO('yolov8n.pt')
# license_plate_detector = YOLO('license_plate_detector.pt')

# # load video
# cap = cv2.VideoCapture('sample_video.mp4')
# vehicles = [2, 3, 5, 7]


# # read frames
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#         # detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])

#         # track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))

#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate

#             # assign license plate to car
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

#             if car_id != -1:

#                 # crop license plate
#                 license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

#                 # process license plate
#                 license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                 _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

#                 # read license plate number
#                 license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

#                 if license_plate_text is not None:
#                     print(f"[INFO] Frame {frame_nmr} | Car ID {car_id} | Plate: {license_plate_text} | Score: {license_plate_text_score}")
#                     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                     'text': license_plate_text,
#                                                                     'bbox_score': score,
#                                                                     'text_score': license_plate_text_score}}
#                 else:
#                     print(f"[WARN] OCR failed for license plate in frame {frame_nmr}")
    
# # write results
# print(f"Total frames with results: {len(results)}")
# write_csv(results, './test.csv')





import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import easyocr
from sort.sort import *
from io import BytesIO
import tempfile
import time

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Initialize MOT tracker
mot_tracker = Sort()

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break
    if foundIt:
        return vehicle_track_ids[car_indx]
    return -1, -1, -1, -1, -1

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if len(text) == 7:  # Assuming 7 characters for a license plate
            return text, score
    return None, None

def process_video(uploaded_video):
    video_bytes = uploaded_video.read()
    video_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(video_path, 'wb') as f:
        f.write(video_bytes)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_nmr = -1
    ret = True

    # Create an empty placeholder to update the frame continuously
    frame_placeholder = st.empty()

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        if ret:
            print(f"Processing frame {frame_nmr}")  # Debugging line

            results = {}
            detections = coco_model(frame)[0]
            detections_ = []

            # Detect vehicles
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in [2, 3, 5, 7]:  # Vehicle classes
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    if license_plate_text:
                        results[frame_nmr] = {
                            car_id: {
                                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                'license_plate': {
                                    'bbox': [x1, y1, x2, y2],
                                    'text': license_plate_text,
                                    'bbox_score': score,
                                    'text_score': license_plate_text_score
                                }
                            }
                        }

            # Display the frame with detections
            for car_id, data in results.get(frame_nmr, {}).items():
                car_bbox = data['car']['bbox']
                plate_bbox = data['license_plate']['bbox']
                plate_text = data['license_plate']['text']

                # Draw car bounding box
                cv2.rectangle(frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (0, 255, 0), 5)
                # Draw license plate bounding box
                cv2.rectangle(frame, (int(plate_bbox[0]), int(plate_bbox[1])), (int(plate_bbox[2]), int(plate_bbox[3])), (0, 0, 255), 5)
                # Put license plate text
                cv2.putText(frame, plate_text, (int(plate_bbox[0]), int(plate_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the placeholder with the current frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Optional: add a small delay to simulate real-time (adjust based on video FPS)
            time.sleep(0.1)  # You can adjust this value for smoother playback

    cap.release()



# Streamlit UI
st.title('Real-time Vehicle and License Plate Detection')

uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.video(uploaded_video)  # Display video file
    st.write("Processing the video...")
    process_video(uploaded_video)
