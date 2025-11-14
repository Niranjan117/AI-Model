"""
Model Training Module for Crop Analysis AI
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import os
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from config import MODEL_CONFIG, PATHS

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['water', 'forest', 'wheat', 'rice', 'corn', 'barren', 'annual_crop', 'permanent_crop']
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(PATHS['MODEL_DIR'], exist_ok=True)
        os.makedirs(PATHS['LOGS_DIR'], exist_ok=True)
        
    def build_advanced_cnn(self):
        """Build advanced CNN architecture with attention mechanism"""
        
        input_layer = layers.Input(shape=MODEL_CONFIG['INPUT_SIZE'])
        
        # Data augmentation layer
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        
        x = augmentation(input_layer)
        
        # Feature extraction backbone with residual connections
        x = self._conv_block(x, 64, (7, 7), strides=2)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 64)
        
        x = self._residual_block(x, 128, downsample=True)
        x = self._residual_block(x, 128)
        
        x = self._residual_block(x, 256, downsample=True)
        x = self._residual_block(x, 256)
        
        x = self._residual_block(x, 512, downsample=True)
        x = self._residual_block(x, 512)
        
        # Attention mechanism
        attention = layers.GlobalAveragePooling2D()(x)
        attention = layers.Dense(512, activation='relu')(attention)
        attention = layers.Dense(x.shape[-1], activation='sigmoid')(attention)
        attention = layers.Reshape((1, 1, x.shape[-1]))(attention)
        
        x = layers.Multiply()([x, attention])
        
        # Global pooling and classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Multi-task outputs
        land_cover_output = layers.Dense(len(self.class_names), activation='softmax', name='land_cover')(x)
        
        # Vegetation health regression output
        vegetation_features = layers.Dense(256, activation='relu')(x)
        vegetation_features = layers.Dropout(0.3)(vegetation_features)
        vegetation_output = layers.Dense(4, activation='sigmoid', name='vegetation_health')(vegetation_features)
        
        # Yield prediction output
        yield_features = layers.Dense(128, activation='relu')(x)
        yield_features = layers.Dropout(0.3)(yield_features)
        yield_output = layers.Dense(1, activation='linear', name='yield_prediction')(yield_features)
        
        model = keras.Model(
            inputs=input_layer, 
            outputs=[land_cover_output, vegetation_output, yield_output]
        )
        
        return model
    
    def _conv_block(self, x, filters, kernel_size, strides=1):
        """Convolutional block with batch normalization"""
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def _residual_block(self, x, filters, downsample=False):
        """Residual block with skip connection"""
        strides = 2 if downsample else 1
        
        # Main path
        y = self._conv_block(x, filters, (3, 3), strides=strides)
        y = layers.Conv2D(filters, (3, 3), padding='same')(y)
        y = layers.BatchNormalization()(y)
        
        # Skip connection
        if downsample or x.shape[-1] != filters:
            x = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(x)
            x = layers.BatchNormalization()(x)
        
        # Add skip connection
        y = layers.Add()([x, y])
        y = layers.Activation('relu')(y)
        
        return y
    
    def load_training_data(self):
        """Load training data from image folders"""
        X_data = []
        y_land_cover = []
        y_vegetation = []
        y_yield = []
        
        # Mapping of folder names to class indices
        folder_mapping = {
            'SeaLake': 0, 'River': 0,  # water
            'Forest': 1, 'HerbaceousVegetation': 1,  # forest
            'Wheat': 2,  # wheat
            'Rice': 3,  # rice
            'Sugercane': 4,  # corn (sugarcane as corn substitute)
            'Highway': 5, 'Industrial': 5, 'Residential': 5, 'Pasture': 5,  # barren
            'AnnualCrop': 6,  # annual_crop
            'PermanentCrop': 7  # permanent_crop
        }
        
        base_path = os.getcwd()
        
        for folder_name, class_idx in folder_mapping.items():
            folder_path = os.path.join(base_path, folder_name)
            
            if os.path.exists(folder_path):
                self.logger.info(f"Loading images from {folder_name}...")
                
                # Get all image files
                image_patterns = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']
                image_files = []
                
                for pattern in image_patterns:
                    image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
                    image_files.extend(glob.glob(os.path.join(folder_path, '**', pattern), recursive=True))
                
                # Load images (limit to prevent memory issues)
                for img_path in image_files[:100]:
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, MODEL_CONFIG['INPUT_SIZE'][:2])
                            img = img.astype(np.float32) / 255.0
                            
                            X_data.append(img)
                            
                            # One-hot encode land cover class
                            land_cover_label = np.zeros(len(self.class_names))
                            land_cover_label[class_idx] = 1
                            y_land_cover.append(land_cover_label)
                            
                            # Generate synthetic vegetation health data
                            vegetation_health = self._generate_vegetation_labels(img, class_idx)
                            y_vegetation.append(vegetation_health)
                            
                            # Generate synthetic yield data
                            yield_value = self._generate_yield_labels(class_idx, vegetation_health)
                            y_yield.append(yield_value)
                            
                    except Exception as e:
                        self.logger.warning(f"Error loading {img_path}: {e}")
                        continue
                
                self.logger.info(f"Loaded {len([f for f in image_files[:100] if os.path.exists(f)])} images from {folder_name}")
        
        return np.array(X_data), np.array(y_land_cover), np.array(y_vegetation), np.array(y_yield)
    
    def _generate_vegetation_labels(self, image, class_idx):
        """Generate vegetation health labels based on image and class"""
        # Calculate basic vegetation indices
        red = image[:,:,0]
        green = image[:,:,1]
        blue = image[:,:,2]
        
        # Simulate NIR
        nir = green * 1.2 + np.random.normal(0, 0.05, green.shape)
        
        # Calculate indices
        ndvi = np.mean(np.where((nir + red) > 0, (nir - red) / (nir + red), 0))
        evi = np.mean(np.where((nir + 6*red - 7.5*blue + 1) > 0,
                              2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1), 0))
        green_coverage = np.mean(green > 0.4)
        vegetation_density = np.mean(ndvi > 0.2) if not np.isnan(ndvi) else 0
        
        # Adjust based on land cover class
        if class_idx in [0]:  # water
            ndvi *= 0.1
            evi *= 0.1
            green_coverage *= 0.2
            vegetation_density *= 0.1
        elif class_idx in [1]:  # forest
            ndvi = max(0.6, ndvi)
            evi = max(0.5, evi)
            green_coverage = max(0.7, green_coverage)
            vegetation_density = max(0.8, vegetation_density)
        elif class_idx in [2, 3, 6]:  # crops
            ndvi = max(0.4, min(0.8, ndvi))
            evi = max(0.3, min(0.7, evi))
        
        return np.array([
            np.clip(ndvi, 0, 1),
            np.clip(evi, 0, 1),
            np.clip(green_coverage, 0, 1),
            np.clip(vegetation_density, 0, 1)
        ])
    
    def _generate_yield_labels(self, class_idx, vegetation_health):
        """Generate yield labels based on class and vegetation health"""
        base_yields = {
            0: 0,    # water
            1: 0,    # forest
            2: 4.5,  # wheat
            3: 6.2,  # rice
            4: 5.8,  # corn/sugarcane
            5: 0,    # barren
            6: 4.0,  # annual_crop
            7: 3.5   # permanent_crop
        }
        
        base_yield = base_yields.get(class_idx, 0)
        
        if base_yield > 0:
            # Adjust yield based on vegetation health
            health_factor = np.mean(vegetation_health)
            yield_value = base_yield * (0.5 + health_factor * 0.8)  # 0.5 to 1.3 multiplier
            yield_value += np.random.normal(0, 0.5)  # Add some noise
            yield_value = max(0, min(15, yield_value))  # Clamp to reasonable range
        else:
            yield_value = 0
        
        return yield_value
    
    def compile_model(self):
        """Compile the model with appropriate loss functions and metrics"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG['LEARNING_RATE']),
            loss={
                'land_cover': 'categorical_crossentropy',
                'vegetation_health': 'mse',
                'yield_prediction': 'mse'
            },
            loss_weights={
                'land_cover': 1.0,
                'vegetation_health': 0.5,
                'yield_prediction': 0.3
            },
            metrics={
                'land_cover': ['accuracy'],
                'vegetation_health': ['mae'],
                'yield_prediction': ['mae']
            }
        )
    
    def train_model(self, X_train, y_land_cover, y_vegetation, y_yield):
        """Train the model with callbacks and validation"""
        
        # Split data
        X_train, X_val, y_lc_train, y_lc_val, y_veg_train, y_veg_val, y_yield_train, y_yield_val = train_test_split(
            X_train, y_land_cover, y_vegetation, y_yield,
            test_size=MODEL_CONFIG['VALIDATION_SPLIT'],
            random_state=42,
            stratify=np.argmax(y_land_cover, axis=1)
        )
        
        # Prepare training data
        train_data = {
            'land_cover': y_lc_train,
            'vegetation_health': y_veg_train,
            'yield_prediction': y_yield_train
        }
        
        val_data = {
            'land_cover': y_lc_val,
            'vegetation_health': y_veg_val,
            'yield_prediction': y_yield_val
        }
        
        # Callbacks
        model_checkpoint = callbacks.ModelCheckpoint(
            os.path.join(PATHS['MODEL_DIR'], 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, train_data,
            validation_data=(X_val, val_data),
            epochs=MODEL_CONFIG['EPOCHS'],
            batch_size=MODEL_CONFIG['BATCH_SIZE'],
            callbacks=[model_checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_land_cover_test, y_vegetation_test, y_yield_test):
        """Evaluate model performance"""
        
        # Make predictions
        predictions = self.model.predict(X_test)
        land_cover_pred, vegetation_pred, yield_pred = predictions
        
        # Land cover classification metrics
        y_true_classes = np.argmax(y_land_cover_test, axis=1)
        y_pred_classes = np.argmax(land_cover_pred, axis=1)
        
        # Classification report
        class_report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Vegetation health metrics
        vegetation_mae = np.mean(np.abs(y_vegetation_test - vegetation_pred))
        vegetation_rmse = np.sqrt(np.mean((y_vegetation_test - vegetation_pred) ** 2))
        
        # Yield prediction metrics
        yield_mae = np.mean(np.abs(y_yield_test - yield_pred.flatten()))
        yield_rmse = np.sqrt(np.mean((y_yield_test - yield_pred.flatten()) ** 2))
        
        evaluation_results = {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'vegetation_mae': vegetation_mae,
            'vegetation_rmse': vegetation_rmse,
            'yield_mae': yield_mae,
            'yield_rmse': yield_rmse
        }
        
        return evaluation_results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            self.logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Land cover accuracy
        axes[0, 1].plot(self.history.history['land_cover_accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_land_cover_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Land Cover Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Vegetation health MAE
        axes[1, 0].plot(self.history.history['vegetation_health_mae'], label='Training MAE')
        axes[1, 0].plot(self.history.history['val_vegetation_health_mae'], label='Validation MAE')
        axes[1, 0].set_title('Vegetation Health MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        
        # Yield prediction MAE
        axes[1, 1].plot(self.history.history['yield_prediction_mae'], label='Training MAE')
        axes[1, 1].plot(self.history.history['val_yield_prediction_mae'], label='Validation MAE')
        axes[1, 1].set_title('Yield Prediction MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS['LOGS_DIR'], f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.show()
    
    def save_model(self, model_name='crop_analysis_model'):
        """Save the trained model"""
        if self.model is None:
            self.logger.error("No model to save")
            return
        
        model_path = os.path.join(PATHS['MODEL_DIR'], f'{model_name}.h5')
        self.model.save(model_path)
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(os.path.join(PATHS['MODEL_DIR'], f'{model_name}_architecture.json'), 'w') as json_file:
            json_file.write(model_json)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def train_complete_pipeline(self):
        """Complete training pipeline"""
        self.logger.info("Starting complete training pipeline...")
        
        # Load data
        self.logger.info("Loading training data...")
        X_data, y_land_cover, y_vegetation, y_yield = self.load_training_data()
        
        if len(X_data) == 0:
            self.logger.error("No training data found!")
            return False
        
        self.logger.info(f"Loaded {len(X_data)} training samples")
        
        # Build and compile model
        self.logger.info("Building model...")
        self.model = self.build_advanced_cnn()
        self.compile_model()
        
        # Print model summary
        self.model.summary()
        
        # Train model
        self.logger.info("Training model...")
        self.train_model(X_data, y_land_cover, y_vegetation, y_yield)
        
        # Save model
        self.save_model()
        
        # Plot training history
        self.plot_training_history()
        
        self.logger.info("Training pipeline completed!")
        return True