from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import cv2
import time
# import numpy as np
import torch
import multiprocessing
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import uvicorn 

app = FastAPI()

pathlib = 'D:\infusion/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('Ultralytics/yolov5', 'custom', path=path, device='0')

# path = '/home/van/Detta/infusion/yolov5/runs/train/exp2/weights/best.pt'
# model = torch.hub.load('Ultralytics/yolov5', 'custom', path=path, device='0')

# Define a model for the input data
class DetectionResult(BaseModel):
    total_drops: int
    long_drops: int

# Initialize variables for drop counting
total_drops_count = 0
start_time = time.time()
last_drop_time = start_time

# Endpoint to count total infusion drops
@app.post("/count_total_infusion_drops/")
def count_total_infusion_drops(detection_result: DetectionResult):
    global total_drops_count
    total_drops_count += detection_result.total_drops
    return {"message": "Total infusion drops counted successfully."}

# Endpoint to count total drops in minutes
@app.get("/count_total_drops_in_minutes/")
def count_total_drops_in_minutes():
    global start_time, last_drop_time, total_drops_count
    current_time = time.time()
    elapsed_time = current_time - start_time
    total_minutes = elapsed_time / 60.0
    drops_per_minute = total_drops_count / total_minutes
    return {"total_drops_count": total_drops_count, "drops_per_minute": drops_per_minute}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
