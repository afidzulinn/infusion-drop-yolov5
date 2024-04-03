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
total_duration = 0
tai = 'bau'

def detect_drops(frame):
    # Perform object detection on the frame
    results = model(frame)
    detections = results.xyxy[0]
    return detections

def count_total_drops(frame):
    detections = detect_drops(frame)
    return len(detections)

def duration_between_drops(frame, last_drop_time):
    detections = detect_drops(frame)
    if detections is not None and len(detections) > 0:
        current_time = time.time()
        if last_drop_time:
            time_diff = current_time - last_drop_time
            return time_diff, current_time
    return None, last_drop_time

def process_frame(frame):
    global total_drops, last_drop_time, total_duration

    # Count total drops
    drop_count = count_total_drops(frame)
    total_drops += drop_count

    # Calculate duration between drops
    duration, last_drop_time = duration_between_drops(frame, last_drop_time)
    if duration is not None:
        total_duration += duration
        # average_duration = total_duration / total_drops
        # print("Total drops:", total_drops)
        # print("Average duration between drops:", average_duration)

@app.post("/start_detection")
async def start_detection(background_tasks: BackgroundTasks):

    global video_capture, total_drops, last_drop_time, total_duration

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Unable to access the camera.")
        return {"message": "Error: Unable to access the camera."}

    total_drops = 0
    last_drop_time = 0
    total_duration = 0

    def detect_objects():
        global video_capture
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            process_frame(frame)
            time.sleep(0.1)

        cv2.destroyAllWindows()

    background_tasks.add_task(detect_objects)

    return {"message": "Object detection started."}
    # def detect_objects():
    #     video_capture = cv2.VideoCapture(0)
    #     if not video_capture.isOpened():
    #         print("Error: Unable to access the camera.")
    #         return
    #
    #     global total_drops, last_drop_time, total_duration
    #     total_drops = 0
    #     last_drop_time = 0
    #     total_duration = 0
    #
    #     while True:
    #         ret, frame = video_capture.read()
    #         if not ret:
    #             print("Error: Unable to capture frame.")
    #             break
    #
    #         process_frame(frame)
    #         time.sleep(0.1)
    #
    #     video_capture.release()
    #     cv2.destroyAllWindows()
    #
    # background_tasks.add_task(detect_objects)
    #
    # return {"message": "Object detection started."}

@app.post("/stop_detection")
async def stop_detection():
    global video_capture

    if video_capture is not None:
        video_capture.release()
        video_capture = None


  # # Get handle to video camera
  # camera = cv2.VideoCapture(0)
  # # Stop camera
  # camera.release()

    # global model
    global total_drops, last_drop_time, total_duration
    total_drops = 0
    last_drop_time = 0
    total_duration = 0

    return {"message": "Object detection and Camera stopped."}

@app.get("/drop_stats")
async def get_drop_stats():
    global total_drops, total_duration

    long_drop_between_drops = total_duration / total_drops if total_drops > 0 else 0
    return {"total_drops": total_drops, "long_drop_between_drops (seconds)": long_drop_between_drops}



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8501, debug=True, reload=True)