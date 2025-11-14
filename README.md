# 🌾 Crop Analysis AI API

An AI-powered crop yield prediction and land use classification system with a REST API for easy website and mobile app integration.

## ✨ Features

* **Crop Yield Prediction**: Predicts yield in tons/hectare with 95%+ accuracy.
* **Land Use Classification**: Identifies percentages of Lake, River, Wheat, Rice, Corn, and Barren land.
* **Vegetation Health Analysis**: Provides NDVI, EVI, green coverage, and vegetation density metrics.
* **REST API**: Simple integration with any website or mobile application.
* **Batch Processing**: Analyze multiple images simultaneously.
* **Auto-sync**: Automatically sends analysis results to your configured endpoints.

## 🚀 Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start Server**:
    ```bash
    python run_server.py
    ```
    **Or use the Windows batch file**:
    ```bash
    start.bat
    ```

3.  **The server will display the accessible URLs**:
    * **Local**: `http://localhost:8000/analyze`
    * **Network**: `http://YOUR_IP:8000/analyze`

    > **Note**: Your network IP address (`YOUR_IP`) will change if you switch to a different WiFi network. You will need to restart the server to get the new IP.

## 🔌 API Endpoints

**Base URL**: `http://YOUR_IP:8000` (replace `YOUR_IP` with the actual IP address shown when the server starts)

---

Here is a clean, attractive version of your README, formatted for GitHub. You can copy and paste the entire content from the box below.

Markdown

# 🌾 Crop Analysis AI API

An AI-powered crop yield prediction and land use classification system with a REST API for easy website and mobile app integration.

## ✨ Features

* **Crop Yield Prediction**: Predicts yield in tons/hectare with 95%+ accuracy.
* **Land Use Classification**: Identifies percentages of Lake, River, Wheat, Rice, Corn, and Barren land.
* **Vegetation Health Analysis**: Provides NDVI, EVI, green coverage, and vegetation density metrics.
* **REST API**: Simple integration with any website or mobile application.
* **Batch Processing**: Analyze multiple images simultaneously.
* **Auto-sync**: Automatically sends analysis results to your configured endpoints.

## 🚀 Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start Server**:
    ```bash
    python run_server.py
    ```
    **Or use the Windows batch file**:
    ```bash
    start.bat
    ```

3.  **The server will display the accessible URLs**:
    * **Local**: `http://localhost:8000/analyze`
    * **Network**: `http://YOUR_IP:8000/analyze`

    > **Note**: Your network IP address (`YOUR_IP`) will change if you switch to a different WiFi network. You will need to restart the server to get the new IP.

## 🔌 API Endpoints

**Base URL**: `http://YOUR_IP:8000` (replace `YOUR_IP` with the actual IP address shown when the server starts)

---

### `GET /analyze`

Get demo crop data. No image upload is required.

```bash
curl "http://YOUR_IP:8000/analyze"
POST /analyze
Analyze a single satellite image.

Bash

curl -X POST "http://YOUR_IP:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@satellite_image.jpg"
Example Response:

JSON

{
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
  "status": "success"
}
POST /batch-analyze
Analyze multiple images in a single request (max 10).

Bash

curl -X POST "http://YOUR_IP:8000/batch-analyze" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
POST /configure-webhooks
Configure external system endpoints for the auto-sync feature.

Bash

curl -X POST "http://YOUR_IP:8000/configure-webhooks" \
  -H "Content-Type: application/json" \
  -d '{"website_url": "[https://your-website.com/api/crop-data](https://your-website.com/api/crop-data)"}'
💻 Integration Examples
Website Integration (JavaScript)
JavaScript

// Example 1: Get demo data (no image)
fetch('http://YOUR_IP:8000/analyze')
  .then(response => response.json())
  .then(data => {
    console.log('Wheat:', data.land_use_percentages.wheat + '%');
    console.log('Rice:', data.land_use_percentages.rice + '%');
  });

// Example 2: Analyze a real image
const formData = new FormData();
formData.append('file', imageFile); // imageFile is from an <input type="file">

fetch('http://YOUR_IP:8000/analyze', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log('Yield:', data.yield_prediction, 'tons/hectare');
    console.log('Land use:', data.land_use_percentages);
  });
Mobile App Integration (Kotlin)
Kotlin

class CropAnalysisService {
    suspend fun analyzeCropImage(imageFile: File): CropAnalysisResult? {
        // See client_examples.py for a Python-based example
    }
}
📊 Data Output Format
The API returns a comprehensive JSON object with the following data:

yield_prediction: Crop yield in tons per hectare.

land_use_percentages: An object containing the percentage of land cover:

wheat: % of wheat crops

rice: % of rice crops

corn: % of corn crops

lake: % of water bodies (lakes)

river: % of water bodies (rivers)

barren: % of barren/unused land

vegetation_health: An object with normalized health indices (0-1 scale).

🔄 Auto-sync to External Systems
After configuring your website and mobile backend URLs via the /configure-webhooks endpoint, the API will automatically POST the full analysis results to both systems every time a new image is successfully analyzed.

📂 File Structure
ai_model.py: Core AI models for prediction and classification.

api_server.py: The FastAPI REST server logic.

run_server.py: Server runner script.

start.bat: Windows startup script.

requirements.txt: Python dependencies.

📈 Model Performance
Yield Prediction Accuracy: 95%+ within ±3 tons/hectare.

Land Use Classification: 90%+ classification accuracy.

Processing Time: ~2-3 seconds per image.

Supported Formats: JPG, PNG, TIFF.

```bash
curl "http://YOUR_IP:8000/analyze"
