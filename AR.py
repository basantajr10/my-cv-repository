from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

obj_model = YOLO("yolo11l.pt")
face_model = YOLO("best.onnx")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"objects": [], "emotions": []}

    frame = cv2.flip(frame, 1)

    # Object Detection
    obj_results = obj_model(frame, conf=0.5, imgsz=640, verbose=False)[0]
    # Face Emotion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.merge([gray, gray, gray])
    face_results = face_model(gray3, conf=0.4, verbose=False)[0]

    objects = []
    for box in obj_results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        label = obj_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        objects.append({"label": label, "conf": round(conf,2), "box": [x1,y1,x2,y2]})

    emotions = []
    for box in face_results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        label = face_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        emotions.append({"label": label, "conf": round(conf,2), "box": [x1,y1,x2,y2]})

    return {"objects": objects, "emotions": emotions}

print("WEB Server Ready → http://127.0.0.1:8000/predict")