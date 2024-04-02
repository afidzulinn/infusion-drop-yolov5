from fastapi import FastAPI, BackgroundTasks
import cv2
import time
# import numpy as np
import torch
import multiprocessing

app = FastAPI()

path = '/home/van/Detta/infusion/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('Ultralytics/yolov5', 'custom', path=path, device='0')

manager = multiprocessing.Manager()
total_drops = manager.Value('i', 0)
last_drop_time = manager.Value('d', None)
total_duration = manager.Value('d', 0)

def detect_drops(frame):
    # Perform object detection on the frame
    results = model(frame)
    detections = results.xyxy[0]  # Extract detected objects
    return detections

def count_total_drops(frame):
    detections = detect_drops(frame)
    return len(detections)

def duration_between_drops(frame, last_drop_time):
    detections = detect_drops(frame)
    if detections is not None and len(detections) > 0:
        current_time = time.time()
        if last_drop_time.value:
            time_diff = current_time - last_drop_time.value
            return time_diff, current_time
    return None, last_drop_time.value

def process_frame(frame):
    global total_drops, last_drop_time, total_duration

    # Count total drops
    drop_count = count_total_drops(frame)
    with total_drops.get_lock():
        total_drops.value += drop_count

    # Calculate duration between drops
    duration, last_drop_time.value = duration_between_drops(frame, last_drop_time.value)
    if duration is not None:
        with total_duration.get_lock():
            total_duration.value += duration
        average_duration = total_duration.value / total_drops.value
        print("Total drops:", total_drops.value)
        print("Average duration between drops:", average_duration)

@app.post("/start_detection")
async def start_detection(background_tasks: BackgroundTasks):
    def detect_objects():
        video_capture = cv2.VideoCapture(0)  # Open default camera (change to your camera index if needed)
        if not video_capture.isOpened():
            print("Error: Unable to access the camera.")
            return

        with total_drops.get_lock():
            total_drops.value = 0
        with last_drop_time.get_lock():
            last_drop_time.value = None
        with total_duration.get_lock():
            total_duration.value = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            process_frame(frame)
            time.sleep(0.1)  # Adjust sleep time if needed for performance

        video_capture.release()
        cv2.destroyAllWindows()

    background_tasks.add_task(detect_objects)

    return {"message": "Object detection started."}

@app.post("/stop_detection")
async def stop_detection():
    # Implement stopping object detection if needed
    return {"message": "Object detection stopped."}

@app.get("/drop_stats")
async def get_drop_stats():
    with total_drops.get_lock(), total_duration.get_lock():
        return {"total_drops": total_drops.value, "long_drop (seconds)": total_duration.value / total_drops.value if total_drops.value > 0 else 0}
