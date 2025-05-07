import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
from collections import Counter

# Dataset paths
DATASET_PATH = os.path.join(os.getcwd(), 'Casiadataset')
IMAGE_PATH = os.path.join(DATASET_PATH, 'IMAGE')
MASK_PATH = os.path.join(DATASET_PATH, 'MASK')

# Authentic and tampered image paths
AU_PATH = os.path.join(IMAGE_PATH, 'Au')
TP_PATH = os.path.join(IMAGE_PATH, 'Tp')

# Mask paths
AU_MASK_PATH = os.path.join(MASK_PATH, 'Au') if os.path.exists(os.path.join(MASK_PATH, 'Au')) else None
TP_MASK_PATH = os.path.join(MASK_PATH, 'Tp')

def analyze_dataset_structure():
    """Analyze the structure of the CASIA dataset"""
    print("\n===== CASIA Dataset Analysis =====")
    
    # Count images in each folder
    au_images = os.listdir(AU_PATH) if os.path.exists(AU_PATH) else []
    tp_images = os.listdir(TP_PATH) if os.path.exists(TP_PATH) else []
    
    # Count masks
    au_masks = os.listdir(AU_MASK_PATH) if AU_MASK_PATH and os.path.exists(AU_MASK_PATH) else []
    tp_masks = os.listdir(TP_MASK_PATH) if os.path.exists(TP_MASK_PATH) else []
    
    print(f"Authentic Images: {len(au_images)}")
    print(f"Tampered Images: {len(tp_images)}")
    print(f"Authentic Masks: {len(au_masks)}")
    print(f"Tampered Masks: {len(tp_masks)}")
    
    # Check if masks match images
    if AU_MASK_PATH:
        au_match = set([os.path.splitext(img)[0] for img in au_images]) == set([os.path.splitext(mask)[0] for mask in au_masks])
        print(f"Authentic images match masks: {au_match}")
    
    tp_match = set([os.path.splitext(img)[0] for img in tp_images]) == set([os.path.splitext(mask)[0] for mask in tp_masks])
    print(f"Tampered images match masks: {tp_match}")
    
    return {
        'au_images': len(au_images),
        'tp_images': len(tp_images),
        'au_masks': len(au_masks),
        'tp_masks': len(tp_masks)
    }

def analyze_image_properties(sample_size=50):
    """Analyze properties of images in the dataset"""
    print("\n===== Image Properties Analysis =====")
    
    # Sample images from authentic and tampered sets
    au_images = os.listdir(AU_PATH) if os.path.exists(AU_PATH) else []
    tp_images = os.listdir(TP_PATH) if os.path.exists(TP_PATH) else []
    
    au_samples = random.sample(au_images, min(sample_size, len(au_images)))
    tp_samples = random.sample(tp_images, min(sample_size, len(tp_images)))
    
    # Analyze dimensions and color channels
    au_dimensions = []
    au_channels = []
    tp_dimensions = []
    tp_channels = []
    
    for img_name in au_samples:
        img_path = os.path.join(AU_PATH, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            au_dimensions.append(img.shape[:2])
            au_channels.append(img.shape[2] if len(img.shape) > 2 else 1)
    
    for img_name in tp_samples:
        img_path = os.path.join(TP_PATH, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            tp_dimensions.append(img.shape[:2])
            tp_channels.append(img.shape[2] if len(img.shape) > 2 else 1)
    
    # Print results
    print("Authentic Images:")
    print(f"  Dimensions: {Counter([dim for dim in au_dimensions]).most_common(3)}")
    print(f"  Channels: {Counter(au_channels).most_common()}")
    
    print("Tampered Images:")
    print(f"  Dimensions: {Counter([dim for dim in tp_dimensions]).most_common(3)}")
    print(f"  Channels: {Counter(tp_channels).most_common()}")
    
    return {
        'au_dimensions': au_dimensions,
        'au_channels': au_channels,
        'tp_dimensions': tp_dimensions,
        'tp_channels': tp_channels
    }

def analyze_mask_properties(sample_size=50):
    """Analyze properties of mask images"""
    print("\n===== Mask Properties Analysis =====")
    
    # Sample masks from authentic and tampered sets
    au_masks = os.listdir(AU_MASK_PATH) if AU_MASK_PATH and os.path.exists(AU_MASK_PATH) else []
    tp_masks = os.listdir(TP_MASK_PATH) if os.path.exists(TP_MASK_PATH) else []
    
    if au_masks:
        au_samples = random.sample(au_masks, min(sample_size, len(au_masks)))
        
        # Analyze authentic masks (should be all black)
        au_mask_values = []
        for mask_name in au_samples:
            mask_path = os.path.join(AU_MASK_PATH, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_values = np.unique(mask)
                au_mask_values.append(tuple(unique_values))
        
        print("Authentic Masks:")
        print(f"  Unique values: {Counter(au_mask_values).most_common()}")
    
    if tp_masks:
        tp_samples = random.sample(tp_masks, min(sample_size, len(tp_masks)))
        
        # Analyze tampered masks (should have white regions on black background)
        tp_mask_values = []
        tp_white_percentage = []
        
        for mask_name in tp_samples:
            mask_path = os.path.join(TP_MASK_PATH, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_values = np.unique(mask)
                tp_mask_values.append(tuple(unique_values))
                
                # Calculate percentage of white pixels (tampered regions)
                if 255 in unique_values:
                    white_percentage = np.sum(mask == 255) / mask.size * 100
                    tp_white_percentage.append(white_percentage)
        
        print("Tampered Masks:")
        print(f"  Unique values: {Counter(tp_mask_values).most_common()}")
        if tp_white_percentage:
            print(f"  Average tampered region: {np.mean(tp_white_percentage):.2f}% of image")
            print(f"  Min tampered region: {np.min(tp_white_percentage):.2f}%")
            print(f"  Max tampered region: {np.max(tp_white_percentage):.2f}%")
    
    return {
        'au_mask_values': au_mask_values if 'au_mask_values' in locals() else None,
        'tp_mask_values': tp_mask_values if 'tp_mask_values' in locals() else None,
        'tp_white_percentage': tp_white_percentage if 'tp_white_percentage' in locals() else None
    }

def visualize_samples():
    """Visualize sample images and their masks"""
    print("\n===== Sample Visualization =====")
    
    # Try to find a matching pair of tampered image and mask
    tp_images = os.listdir(TP_PATH) if os.path.exists(TP_PATH) else []
    tp_masks = os.listdir(TP_MASK_PATH) if os.path.exists(TP_MASK_PATH) else []
    
    # Get a random tampered image
    if tp_images and tp_masks:
        tp_img_name = random.choice(tp_images)
        tp_img_base = os.path.splitext(tp_img_name)[0]
        
        # Find corresponding mask
        matching_mask = None
        for mask_name in tp_masks:
            mask_base = os.path.splitext(mask_name)[0]
            if mask_base == tp_img_base:
                matching_mask = mask_name
                break
        
        if matching_mask:
            # Load and display the image and mask
            tp_img_path = os.path.join(TP_PATH, tp_img_name)
            tp_mask_path = os.path.join(TP_MASK_PATH, matching_mask)
            
            tp_img = cv2.imread(tp_img_path)
            tp_mask = cv2.imread(tp_mask_path, cv2.IMREAD_GRAYSCALE)
            
            if tp_img is not None and tp_mask is not None:
                # Convert BGR to RGB for display
                tp_img_rgb = cv2.cvtColor(tp_img, cv2.COLOR_BGR2RGB)
                
                # Create a visualization of the tampered region
                highlighted = tp_img_rgb.copy()
                highlighted[tp_mask == 255] = [255, 0, 0]  # Highlight tampered regions in red
                
                # Display
                plt.figure(figsize=(15, 5))
                plt.subplot(131)
                plt.imshow(tp_img_rgb)
                plt.title('Tampered Image')
                plt.axis('off')
                
                plt.subplot(132)
                plt.imshow(tp_mask, cmap='gray')
                plt.title('Mask (Tampered Regions)')
                plt.axis('off')
                
                plt.subplot(133)
                plt.imshow(highlighted)
                plt.title('Highlighted Tampered Regions')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig('sample_visualization.png')
                print("Sample visualization saved as 'sample_visualization.png'")
            else:
                print("Could not load sample image or mask")
        else:
            print("Could not find matching mask for tampered image")
    else:
        print("No tampered images or masks found")

def main():
    """Main function to run all analyses"""
    print("Starting CASIA dataset analysis...")
    
    # Run analyses
    structure = analyze_dataset_structure()
    image_props = analyze_image_properties()
    mask_props = analyze_mask_properties()
    
    # Visualize samples
    visualize_samples()
    
    print("\n===== Analysis Complete =====")
    print("Dataset is ready for model development")

if __name__ == "__main__":
    main()