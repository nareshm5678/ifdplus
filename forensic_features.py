import cv2
import numpy as np
from scipy import ndimage
from skimage import feature
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from PIL import Image, ImageChops, ImageEnhance
import torch
from torchvision import transforms

class ForensicFeatureExtractor:
    """Extract traditional forensic features from images for forgery detection"""
    
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def extract_all_features(self, image_path):
        """Extract all forensic features from an image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Extract features
        ela_features = self.extract_ela_features(image)
        noise_features = self.extract_noise_features(image_np)
        texture_features = self.extract_texture_features(image_np)
        metadata_features = self.extract_metadata_features(image)
        
        # Combine all features
        combined_features = np.concatenate([
            ela_features.flatten(),
            noise_features.flatten(),
            texture_features.flatten(),
            metadata_features.flatten()
        ])
        
        # Convert to tensor
        return torch.FloatTensor(combined_features)
    
    def extract_ela_features(self, image, quality=90):
        """Extract Error Level Analysis features"""
        # Save image with specific quality
        temp_path = 'temp_ela.jpg'
        image.save(temp_path, 'JPEG', quality=quality)
        
        # Load compressed image
        compressed_image = Image.open(temp_path)
        
        # Calculate ELA
        ela_image = ImageChops.difference(image, compressed_image)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Convert to tensor and resize
        ela_tensor = self.transform(ela_image)
        
        # Remove temporary file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Return features
        return ela_tensor.numpy()
    
    def extract_noise_features(self, image_np):
        """Extract noise features using noise residuals"""
        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply denoising filter
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Calculate noise residual
        noise = gray.astype(np.float32) - denoised.astype(np.float32)
        
        # Normalize noise
        noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Resize to standard size
        noise = cv2.resize(noise, (self.img_size, self.img_size))
        
        # Return features
        return noise.reshape(1, self.img_size, self.img_size)
    
    def extract_texture_features(self, image_np):
        """Extract texture features using Local Binary Patterns"""
        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Resize to standard size
        gray = cv2.resize(gray, (self.img_size, self.img_size))
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Normalize LBP
        lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Return features
        return lbp.reshape(1, self.img_size, self.img_size)
    
    def extract_metadata_features(self, image):
        """Extract metadata-based features"""
        # This is a placeholder for metadata extraction
        # In a real implementation, you would extract EXIF data, compression artifacts, etc.
        # For now, we'll return a dummy feature vector
        return np.zeros((1, self.img_size, self.img_size))

# Function to extract features from a batch of images
def extract_batch_features(image_paths, img_size=256):
    """Extract forensic features from a batch of images"""
    extractor = ForensicFeatureExtractor(img_size=img_size)
    features = []
    
    for path in image_paths:
        feature_vector = extractor.extract_all_features(path)
        features.append(feature_vector)
    
    return torch.stack(features)

# Function to visualize ELA
def visualize_ela(image_path, quality=90, save_path=None):
    """Visualize Error Level Analysis"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Save image with specific quality
    temp_path = 'temp_ela.jpg'
    image.save(temp_path, 'JPEG', quality=quality)
    
    # Load compressed image
    compressed_image = Image.open(temp_path)
    
    # Calculate ELA
    ela_image = ImageChops.difference(image, compressed_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Save or return
    if save_path:
        ela_image.save(save_path)
        print(f"ELA visualization saved to {save_path}")
        
        # Remove temporary file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return save_path
    else:
        # Remove temporary file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return ela_image