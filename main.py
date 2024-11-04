from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
from ultralytics import YOLO
from PIL import Image
import uuid

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://saroperation.netlify.app"],  # Specify your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and results
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load YOLOv8 model
model = YOLO("model/best_sar_model.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Generate unique filename
    file_id = str(uuid.uuid4())
    input_path = f"uploads/{file_id}.jpg"
    output_path = f"results/{file_id}.jpg"
    
    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run inference
    results = model(input_path)
    
    # Save result image
    result_img = Image.fromarray(results[0].plot())
    result_img.save(output_path)
    
    # Schedule cleanup of both input and output files after response is sent
    background_tasks.add_task(cleanup_files, input_path, output_path)
    
    # Return the result image
    return FileResponse(output_path)

def cleanup_files(*paths):
    """Delete files from given paths."""
    for path in paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
