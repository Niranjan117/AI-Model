from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
import time
from ai_model import CropAnalysisAI
import requests

app = FastAPI(title="Crop Analysis AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI model instance
ai_model = None
WEBSITE_ENDPOINT = None
MOBILE_ENDPOINT = None

class WebhookConfig(BaseModel):
    website_url: str = None
    mobile_url: str = None

@app.on_event("startup")
async def startup_event():
    global ai_model
    print("Loading Crop Analysis AI...")
    ai_model = CropAnalysisAI()
    ai_model.load_models()
    print("AI models ready")

@app.get("/")
async def root():
    return {"message": "Crop Analysis AI API", "status": "running", "version": "1.0.0"}

@app.post("/analyze")
async def analyze_crop_image(file: UploadFile = File(...)):
    """Analyze single uploaded image - REAL PROCESSING ONLY"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    print(f"Starting analysis of: {file.filename}")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # REAL ANALYSIS - takes time
        results = ai_model.analyze_image(tmp_file_path)
        os.unlink(tmp_file_path)
        
        processing_time = round(time.time() - start_time, 2)
        print(f"Analysis completed in {processing_time} seconds")
        
        response_data = {
            "filename": file.filename,
            "processing_time_seconds": processing_time,
            "yield_prediction": results["yield_prediction"],
            "land_use_percentages": results["land_use_percentages"],
            "vegetation_health": results["vegetation_health"],
            "status": "analyzed"
        }
        
        return response_data
        
    except Exception as e:
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analyze")
async def get_demo_info():
    """Info endpoint - no fake data"""
    return {
        "message": "Upload an image using POST to get real analysis",
        "instructions": "Send POST request with 'file' parameter containing image",
        "supported_formats": ["jpg", "jpeg", "png", "tiff"],
        "status": "ready"
    }

@app.post("/batch-analyze")
async def batch_analyze(files: list[UploadFile] = File(...)):
    """Analyze multiple images - REAL PROCESSING"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    start_time = time.time()
    results = []
    
    for i, file in enumerate(files):
        if not file.content_type.startswith('image/'):
            results.append({"filename": file.filename, "error": "Not an image file"})
            continue
            
        try:
            print(f"Processing image {i+1}/{len(files)}: {file.filename}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            analysis = ai_model.analyze_image(tmp_file_path)
            analysis["filename"] = file.filename
            results.append(analysis)
            
            os.unlink(tmp_file_path)
            
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    total_time = round(time.time() - start_time, 2)
    
    return {
        "total_processing_time_seconds": total_time,
        "images_processed": len([r for r in results if "error" not in r]),
        "results": results,
        "status": "batch_completed"
    }

@app.post("/configure-webhooks")
async def configure_webhooks(config: WebhookConfig):
    global WEBSITE_ENDPOINT, MOBILE_ENDPOINT
    
    if config.website_url:
        WEBSITE_ENDPOINT = config.website_url
    if config.mobile_url:
        MOBILE_ENDPOINT = config.mobile_url
    
    return {"message": "Endpoints updated", "website": WEBSITE_ENDPOINT, "mobile": MOBILE_ENDPOINT}