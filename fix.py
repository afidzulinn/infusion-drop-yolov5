import os
from fastapi import FastAPI, BackgroundTasks
import cv2
import time
import torch
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

device = '0' if torch.cuda.is_available() else 'cpu'
pathlib = Path('D:\infusion/yolov5/runs/train/exp2/weights/best.pt')
# path = 'D:/infusion/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('Ultralytics/yolov5', 'custom', path=pathlib, device=device, force_reload=True)

video_capture = None
last_drop_time = 0
total_drops = 0
drops_in_one_minute = 0
start_time = time.time()

def detect_drops(frame):
    results = model(frame)
    detections = results.xyxy[0]
    return detections

def count_total_drops(frame):
    detections = detect_drops(frame)
    return len(detections)

def process_frame(frame):
    global total_drops, last_drop_time, drops_in_one_minute, start_time

    drop_count = count_total_drops(frame)
    total_drops += drop_count

    current_time = time.time()
    time_diff = current_time - start_time
    if time_diff >= 60:
        drops_in_one_minute = total_drops - drops_in_one_minute
        start_time = current_time

@app.post("/start_detection")
async def start_detection(background_tasks: BackgroundTasks):
    global video_capture, total_drops, last_drop_time, drops_in_one_minute, start_time

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return {"message": "Error: Unable to access the camera."}

    total_drops = 0
    drops_in_last_minute = 0
    start_time = time.time()

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

    global total_drops, drops_in_one_minute, start_time
    total_drops = 0
    drops_in_one_minute = 0
    start_time = time.time()

    return {"message": "Object detection and camera stopped."}

@app.get("/drop_stats")
async def get_drop_stats():
    global total_drops, drops_in_last_minute

    return {"total_drops": total_drops, "drops_in_one_minute": drops_in_one_minute}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8501, debug=True, reload=True)
