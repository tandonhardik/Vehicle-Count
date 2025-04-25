import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

import importlib.util
import sys


# Path to sort.py
sort_path = r"C:\Users\tando\OneDrive\Desktop\New folder\sort.py"

# Module name 
module_name = "sort"

# Load the module
spec = importlib.util.spec_from_file_location(module_name, sort_path)
sort_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sort_module)

# Now you can use the Sort class
sort_tracker = sort_module.Sort()


# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8x.pt for best accuracy
class_names = model.model.names

# Video input
video_path = r"C:\Users\tando\OneDrive\Desktop\New folder\Sample-Recording-14mins - Trim.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# SORT tracker
tracker = sort_tracker

# Line position
line_y = frame_height * 2 // 3

# Output video
out = cv2.VideoWriter("output_sort.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

track_cy_history = {}

# Counting
counted_ids = set()
in_count, out_count = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, classes=[2, 3, 5, 7], device='cpu')[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

    # Update tracker
    tracks = tracker.update(np.array(detections))

    # Draw line
    cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 255), 2)

    # for track in tracks:
    #     x1, y1, x2, y2, track_id = map(int, track)
    #     cx = int((x1 + x2) / 2)
    #     cy = int((y1 + y2) / 2)

    #     # Draw box and ID
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     if track_id not in counted_ids:
    #         if cy < line_y + 10 and cy > line_y - 10:
    #             if cx < frame_width // 2:
    #                 in_count += 1
    #             else:
    #                 out_count += 1
    #             counted_ids.add(track_id)

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count based on movement across the line
        if track_id in track_cy_history:
            prev_cy = track_cy_history[track_id]

            if prev_cy < line_y and cy >= line_y and track_id not in counted_ids:
                in_count += 1
                counted_ids.add(track_id)

            elif prev_cy > line_y and cy <= line_y and track_id not in counted_ids:
                out_count += 1
                counted_ids.add(track_id)

        track_cy_history[track_id] = cy


    # Display counts
    cv2.putText(frame, f'IN: {in_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(frame, f'OUT: {out_count}', (frame_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    out.write(frame)
    # cv2.imshow('SORT Tracker', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()

