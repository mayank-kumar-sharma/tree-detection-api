from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import torch
import json
import zipfile

app = FastAPI()

model = YOLO("detection.pt")  # Load your trained detection model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def draw_bounding_boxes(image: Image.Image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run detection
    results = model.predict(image, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    # Draw bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image.copy(), boxes)
    img_byte_arr = io.BytesIO()
    image_with_boxes.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Build JSON result
    json_data = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        width, height = x2 - x1, y2 - y1
        area = width * height

        if area <= 10000:
            size = "Small"
            maturity = "Likely Young"
            co2 = 10
        elif area <= 20000:
            size = "Medium"
            maturity = "Semi-Mature"
            co2 = 20
        else:
            size = "Large"
            maturity = "Mature"
            co2 = 30

        json_data.append({
            "bbox": [x1, y1, x2, y2],
            "canopy_area": int(area),
            "size": size,
            "maturity": maturity,
            "estimated_CO2_kg": co2
        })

    summary = {
        "total_trees": len(json_data),
        "total_canopy_area": sum(obj["canopy_area"] for obj in json_data),
        "total_CO2_kg": sum(obj["estimated_CO2_kg"] for obj in json_data),
        "trees": json_data
    }

    # Prepare ZIP file containing image + JSON
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode="w") as zf:
        zf.writestr("output.jpg", img_byte_arr.getvalue())
        zf.writestr("results.json", json.dumps(summary, indent=4))
    zip_io.seek(0)

    return StreamingResponse(zip_io, media_type="application/x-zip-compressed", headers={
        "Content-Disposition": "attachment; filename=tree_detection_output.zip"
    })






