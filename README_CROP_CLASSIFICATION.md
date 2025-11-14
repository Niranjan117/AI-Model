# Crop Classification System

This system modifies your existing land use dataset to identify specific crops: **Wheat**, **Rice**, and **Sugarcane**.

## Dataset Overview

- **Original dataset**: 19,317 land use images (10 classes)
- **Crop dataset**: 3,000 crop images (3 classes)
  - Wheat: 1,092 images
  - Rice: 1,164 images  
  - Sugarcane: 744 images
- **Combined dataset**: 22,317 images (13 classes)

## Files Created

### Core Scripts
- `crop_dataset_modifier.py` - Creates crop-specific datasets
- `crop_classifier.py` - CNN model for crop classification
- `train_crop_model.py` - Training script
- `predict_crop.py` - Prediction script
- `run_crop_analysis.py` - Complete setup script

### Generated Files
- `crop_train.csv` - Crop-only dataset
- `combined_train.csv` - All classes combined
- `crop_label_map.json` - Crop class mappings
- `combined_label_map.json` - All class mappings
- `crop_classifier_model.h5` - Trained model (after training)

## Usage

### 1. Setup Dataset
```bash
python run_crop_analysis.py
```

### 2. Train Model
```bash
python train_crop_model.py
```

### 3. Make Predictions

**Single image:**
```bash
python predict_crop.py "path/to/image.jpg"
```

**Test samples:**
```bash
python predict_crop.py
```

## Model Architecture

- **Input**: 64x64 RGB images
- **Architecture**: CNN with 3 Conv2D layers
- **Classes**: 3 (Wheat=0, Rice=1, Sugarcane=2)
- **Output**: Crop type with confidence scores

## Label Mappings

### Crop-Only (3 classes)
- 0: Wheat
- 1: Rice  
- 2: Sugarcane

### Combined Dataset (13 classes)
- 0-9: Original land use classes
- 10: Wheat
- 11: Rice
- 12: Sugarcane

## Example Usage

```python
from crop_classifier import CropClassifier

# Initialize classifier
classifier = CropClassifier()

# Predict crop type
result = classifier.predict_crop('path/to/crop_image.jpg')

print(f"Crop: {result['crop']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Requirements

```
tensorflow
opencv-python
pandas
numpy
scikit-learn
```

## Next Steps

1. **Add more crops**: Create new folders and update the mapping
2. **Improve accuracy**: Add data augmentation, try different architectures
3. **Deploy model**: Create web interface or mobile app
4. **Real-time detection**: Use with camera feed for field analysis

## Notes

- Images are automatically resized to 64x64 pixels
- Model uses RGB color space
- Training includes 20% validation split
- Supports both .jpg and .tif image formats