"""
Data Processing Module for Satellite Image Analysis
"""
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from config import MODEL_CONFIG, THRESHOLDS

class SatelliteImageProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        
    def preprocess_image(self, image_path, target_size=(256, 256)):
        """Preprocess satellite image for analysis"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert color space
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Apply histogram equalization for better contrast
            image = self._enhance_contrast(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32) / 255.0
    
    def extract_spectral_features(self, image):
        """Extract spectral features from satellite image"""
        red = image[:,:,0]
        green = image[:,:,1]
        blue = image[:,:,2]
        
        # Simulate NIR band (in real satellites, this would be actual NIR)
        nir = green * 1.3 + np.random.normal(0, 0.02, green.shape)
        nir = np.clip(nir, 0, 1)
        
        features = {}
        
        # Vegetation Indices
        features['ndvi'] = self._calculate_ndvi(nir, red)
        features['evi'] = self._calculate_evi(nir, red, blue)
        features['savi'] = self._calculate_savi(nir, red)
        features['gndvi'] = self._calculate_gndvi(nir, green)
        features['ndwi'] = self._calculate_ndwi(green, nir)
        
        # Spectral ratios
        features['red_green_ratio'] = np.mean(red / (green + 1e-8))
        features['nir_red_ratio'] = np.mean(nir / (red + 1e-8))
        features['blue_green_ratio'] = np.mean(blue / (green + 1e-8))
        
        # Texture features
        features.update(self._calculate_texture_features(image))
        
        return features
    
    def _calculate_ndvi(self, nir, red):
        """Calculate Normalized Difference Vegetation Index"""
        ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0)
        return np.clip(ndvi, -1, 1)
    
    def _calculate_evi(self, nir, red, blue):
        """Calculate Enhanced Vegetation Index"""
        evi = np.where((nir + 6*red - 7.5*blue + 1) > 0,
                      2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1), 0)
        return np.clip(evi, -1, 1)
    
    def _calculate_savi(self, nir, red, L=0.5):
        """Calculate Soil Adjusted Vegetation Index"""
        savi = np.where((nir + red + L) > 0, 
                       (1 + L) * (nir - red) / (nir + red + L), 0)
        return np.clip(savi, -1, 1)
    
    def _calculate_gndvi(self, nir, green):
        """Calculate Green Normalized Difference Vegetation Index"""
        gndvi = np.where((nir + green) > 0, (nir - green) / (nir + green), 0)
        return np.clip(gndvi, -1, 1)
    
    def _calculate_ndwi(self, green, nir):
        """Calculate Normalized Difference Water Index"""
        ndwi = np.where((green + nir) > 0, (green - nir) / (green + nir), 0)
        return np.clip(ndwi, -1, 1)
    
    def _calculate_texture_features(self, image):
        """Calculate texture features using GLCM"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'gradient_mean': np.mean(magnitude),
            'gradient_std': np.std(magnitude),
            'edge_density': edge_density,
            'texture_contrast': np.std(gray),
            'texture_homogeneity': 1 / (1 + np.var(gray))
        }
    
    def segment_land_cover(self, image, n_clusters=8):
        """Segment image into land cover classes using K-means"""
        # Reshape image for clustering
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pixels)
        
        # Reshape back to image dimensions
        segmented = cluster_labels.reshape(image.shape[:2])
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            mask = segmented == i
            cluster_stats[i] = {
                'percentage': np.sum(mask) / mask.size * 100,
                'mean_color': np.mean(pixels[cluster_labels == i], axis=0)
            }
        
        return segmented, cluster_stats
    
    def detect_field_boundaries(self, image):
        """Detect agricultural field boundaries"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = image.shape[0] * image.shape[1] * 0.01  # 1% of image area
        field_boundaries = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return field_boundaries
    
    def calculate_field_statistics(self, image, boundaries):
        """Calculate statistics for detected fields"""
        field_stats = []
        
        for i, boundary in enumerate(boundaries):
            # Create mask for this field
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [boundary], 255)
            
            # Extract field pixels
            field_pixels = image[mask > 0]
            
            if len(field_pixels) > 0:
                stats = {
                    'field_id': i,
                    'area_pixels': np.sum(mask > 0),
                    'mean_rgb': np.mean(field_pixels, axis=0),
                    'std_rgb': np.std(field_pixels, axis=0),
                    'vegetation_score': self._calculate_vegetation_score(field_pixels)
                }
                field_stats.append(stats)
        
        return field_stats
    
    def _calculate_vegetation_score(self, pixels):
        """Calculate vegetation health score for field pixels"""
        if len(pixels) == 0:
            return 0
        
        # Simple vegetation score based on green dominance
        red = pixels[:, 0]
        green = pixels[:, 1]
        blue = pixels[:, 2]
        
        # Green vegetation typically has high green, moderate red, low blue
        green_dominance = np.mean(green > red) * np.mean(green > blue)
        green_intensity = np.mean(green)
        
        vegetation_score = (green_dominance * 0.6 + green_intensity * 0.4)
        return np.clip(vegetation_score, 0, 1)