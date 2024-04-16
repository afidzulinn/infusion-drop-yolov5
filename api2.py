import os
# import sys
# sys.path.insert(0, './yolov5')
from fastapi import FastAPI, BackgroundTasks
import cv2
import models
import time
import torch
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

device = '0' if torch.cuda.is_available() else 'cpu'
pathlib = 'model/infusion_drop.pt'
model = torch.load(pathlib, map_location=device)
# model = torch.load('infusion-drop.pt', map_location=device)

video_capture = None
last_drop_time = 0
total_drops = 0
total_duration = 0

def detect_drops(frame):
    results = model(frame)
    detections = results.xyxy[0]
    return detections

def count_total_drops(frame):
    detections = detect_drops(frame)
    return len(detections)

def process_frame(frame):
    global total_drops, last_drop_time, total_duration

    drop_count = count_total_drops(frame)
    total_drops += drop_count

    current_time = time.time()
    if last_drop_time:
        time_diff = current_time - last_drop_time
        if drop_count > 0:
            total_duration += time_diff

    last_drop_time = current_time

@app.post("/start_detection")
async def start_detection(background_tasks: BackgroundTasks):
    global video_capture, total_drops, last_drop_time, total_duration

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return {"message": "Error: Unable to access the camera."}

    total_drops = 0
    last_drop_time = 0
    total_duration = 0

    def detect_objects():
        global video_capture
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            process_frame(frame)
            time.sleep(0.1)

        cv2.destroyAllWindows()

    background_tasks.add_task(detect_objects)

    return {"message": "Object detection started."}

@app.post("/stop_detection")
async def stop_detection():
    global video_capture

    if video_capture is not None:
        video_capture.release()
        video_capture = None

    return {"message": "Object detection and camera stopped."}

@app.get("/drop_stats")
async def get_drop_stats():
    global total_drops, total_duration

    avg_time_per_drop = total_duration / total_drops if total_drops > 0 else 0
    return {"total_drops": total_drops, "avg_time_per_drop (minutes)": avg_time_per_drop / 60}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8501, debug=True, reload=True)
