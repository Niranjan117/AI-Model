# Agricultural Analysis API

A machine learning-powered API for crop yield prediction and land use classification from satellite imagery.

## Features

* **Crop Yield Prediction**: Estimates agricultural yield in tons per hectare
* **Land Use Classification**: Identifies different land types including crops, water bodies, and barren land
* **Vegetation Health Analysis**: Calculates NDVI, EVI, and vegetation density metrics
* **REST API**: Easy integration with web and mobile applications
* **Batch Processing**: Analyze multiple images simultaneously

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Server**:
   ```bash
   python run_server.py
   ```

3. **API Endpoints**:
   * `POST /analyze` - Analyze single image
   * `POST /batch-analyze` - Analyze multiple images
   * `GET /` - API information

## API Usage

### Single Image Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@satellite_image.jpg"
```

### Batch Analysis
```bash
curl -X POST "http://localhost:8000/batch-analyze" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

## Response Format Example

```json
{
  "filename": "satellite_image.jpg",
  "yield_prediction": 18.5,
  "land_use_percentages": {
    "wheat": 45.2,
    "rice": 23.1,
    "corn": 15.8,
    "lake": 8.3,
    "river": 4.1,
    "barren": 3.5
  },
  "vegetation_health": {
    "ndvi": 0.72,
    "evi": 0.68,
    "green_coverage": 0.81,
    "vegetation_density": 0.75
  },
  "status": "analyzed"
}
```

## Deployment

The API can be deployed on cloud platforms like Render, Heroku, or AWS using the included configuration files.

## Technical Details

* **Framework**: FastAPI
* **ML Libraries**: TensorFlow, OpenCV, scikit-learn
* **Image Processing**: Computer vision algorithms for land classification
* **Supported Formats**: JPG, PNG, TIFF