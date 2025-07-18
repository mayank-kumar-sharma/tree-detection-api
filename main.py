from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from collections import defaultdict
from io import BytesIO
import json
from typing import Optional
from shapely.geometry import Point, Polygon

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO("deetection.pt")  # Ensure this file is present

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    polygon_json: Optional[str] = Form(None)
):
    try:
        # Load and prepare image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Optional polygon mask
        if polygon_json:
            try:
                polygon_points = json.loads(polygon_json)
                polygon = Polygon(polygon_points)
                mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                for y in range(mask.shape[0]):
                    for x in range(mask.shape[1]):
                        if polygon.contains(Point(x, y)):
                            mask[y, x] = 255
                image_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid polygon format: {e}"})

        # Save temp image
        temp_path = "temp_input.jpg"
        result_img_path = "result.jpg"
        cv2.imwrite(temp_path, image_bgr)

        # Run detection
        results = model(temp_path)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        # Annotate image
        annotated_image = image_bgr.copy()

        # Output vars
        output_data = []
        canopy_areas = []
        co2_total = 0
        class_counts = defaultdict(int)

        co2_map = {"S": 10, "M": 20, "L": 30}
        maturity_map = {"S": "likely young", "M": "semi-mature", "L": "mature"}

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box

            if polygon_json:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if mask[cy, cx] == 0:
                    continue

            bbox_area = (x2 - x1) * (y2 - y1)
            size_class = "L" if bbox_area > 20000 else "M" if bbox_area > 10000 else "S"
            co2 = co2_map[size_class]
            maturity = maturity_map[size_class]

            co2_total += co2
            class_counts[size_class] += 1
            canopy_areas.append(bbox_area)

            output_data.append({
                "Tree #": i + 1,
                "Size": size_class,
                "Maturity": maturity,
                "CO2 (kg/year)": co2,
                "Canopy Area (px^2)": int(bbox_area)
            })

            # Draw box and label
            color = (0, 255, 0) if size_class == "S" else (0, 165, 255) if size_class == "M" else (0, 0, 255)
            label = f"{size_class}, {co2}kg"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save result image
        cv2.imwrite(result_img_path, annotated_image)

        # Clean up temp input
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return JSONResponse(content={
            "total_trees": len(output_data),
            "total_co2_kg_per_year": co2_total,
            "average_canopy_area": round(np.mean(canopy_areas), 2) if canopy_areas else 0,
            "class_distribution": dict(class_counts),
            "trees": output_data,
            "result_image": "/result.jpg"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/result.jpg")
def get_result_image():
    if os.path.exists("result.jpg"):
        return FileResponse("result.jpg", media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "No result image found."})

