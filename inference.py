import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import argparse
from model_architecture import ForgeryDetectionNet

def load_model(model_path, device='cuda'):
    """Load a trained forgery detection model"""
    model = ForgeryDetectionNet(num_classes=2, pretrained=False, with_segmentation=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_size=256):
    """Preprocess an image for inference"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Apply transformation
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_image

def predict(model, image_tensor, device='cuda'):
    """Run inference on an image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs, segmentation, attention_maps = model(image_tensor)
        
        # Get classification result
        probabilities = F.softmax(outputs, dim=1)
        prob_authentic, prob_tampered = probabilities[0].cpu().numpy()
        prediction = torch.argmax(outputs, dim=1).item()
        
        # Get segmentation mask and binarize it to hard white/black mask
        segmentation_mask = segmentation[0].cpu().numpy().squeeze()
        segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Resize and combine attention maps
        # Get the size of the first attention map to use as reference
        ref_size = attention_maps[0][0].shape[-2:]
        resized_attention_maps = []
        
        for att in attention_maps:
            # Resize if dimensions don't match
            if att[0].shape[-2:] != ref_size:
                resized_att = F.interpolate(att, size=ref_size, mode='bilinear', align_corners=True)
                resized_attention_maps.append(resized_att[0].unsqueeze(0))
            else:
                resized_attention_maps.append(att[0].unsqueeze(0))
        
        # Combine the resized attention maps
        combined_attention = torch.mean(torch.cat(resized_attention_maps, dim=0), dim=0)
        attention_map = combined_attention.cpu().numpy()
        
    return {
        'prediction': prediction,  # 0 for authentic, 1 for tampered
        'prob_authentic': prob_authentic,
        'prob_tampered': prob_tampered,
        'segmentation_mask': segmentation_mask,
        'attention_map': attention_map
    }

def visualize_prediction(original_image, results, output_path=None):
    """Visualize prediction results"""
    # Get results
    prediction = results['prediction']
    prob_authentic = results['prob_authentic']
    prob_tampered = results['prob_tampered']
    segmentation_mask = results['segmentation_mask'].squeeze()
    attention_map = results['attention_map'].squeeze()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(original_image)
    label = "Authentic" if prediction == 0 else "Tampered"
    confidence = prob_authentic if prediction == 0 else prob_tampered
    axes[0].set_title(f"Prediction: {label} ({confidence:.2f})")
    axes[0].axis('off')
    
    # Plot segmentation mask
    axes[1].imshow(segmentation_mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # Plot attention map
    axes[2].imshow(attention_map, cmap='jet')
    axes[2].set_title("Attention Map")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def overlay_segmentation(original_image, segmentation_mask, threshold=0.5, alpha=0.5, color=[255, 0, 0]):
    """Overlay segmentation mask on the original image"""
    # Resize mask to match original image if needed
    if original_image.shape[:2] != segmentation_mask.shape[:2]:
        segmentation_mask = cv2.resize(segmentation_mask, (original_image.shape[1], original_image.shape[0]))
    
    # Create binary mask
    binary_mask = (segmentation_mask > threshold).astype(np.uint8)
    
    # Create white overlay for tampered regions
    overlay = np.zeros_like(original_image)
    overlay[binary_mask == 1] = [255, 255, 255]
    # Combine with original image: tampered regions show white mask
    combined = original_image.copy()
    combined[binary_mask == 1] = overlay[binary_mask == 1]
    return combined

def process_image(image_path, model_path, output_dir=None, device='cuda'):
    """Process a single image and visualize results"""
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model
    model = load_model(model_path, device)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    
    # Run inference
    results = predict(model, image_tensor, device)
    
    # Determine output path
    if output_dir:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_result.png")
    else:
        output_path = None
    
    # Visualize results
    visualize_prediction(original_image, results, output_path)
    
    # Create overlay image with segmentation mask
    if results['prediction'] == 1:  # If tampered
        overlay = overlay_segmentation(
            original_image, 
            cv2.resize(results['segmentation_mask'].squeeze(), (original_image.shape[1], original_image.shape[0]))
        )
        
        if output_dir:
            overlay_path = os.path.join(output_dir, f"{name}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Overlay saved to {overlay_path}")
    
    return results

def batch_process(image_dir, model_path, output_dir, device='cuda'):
    """Process all images in a directory"""
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model once
    model = load_model(model_path, device)
    
    # Process each image
    results = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing {image_file}...")
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_path)
        
        # Run inference
        result = predict(model, image_tensor, device)
        results.append((image_file, result))
        
        # Determine output path
        name, ext = os.path.splitext(image_file)
        output_path = os.path.join(output_dir, f"{name}_result.png")
        
        # Visualize results
        visualize_prediction(original_image, result, output_path)
        
        # Create overlay image with segmentation mask
        if result['prediction'] == 1:  # If tampered
            overlay = overlay_segmentation(
                original_image, 
                cv2.resize(result['segmentation_mask'].squeeze(), (original_image.shape[1], original_image.shape[0]))
            )
            
            overlay_path = os.path.join(output_dir, f"{name}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Create summary report
    authentic_count = sum(1 for _, r in results if r['prediction'] == 0)
    tampered_count = sum(1 for _, r in results if r['prediction'] == 1)
    
    print("\nSummary:")
    print(f"Total images processed: {len(results)}")
    print(f"Authentic images: {authentic_count}")
    print(f"Tampered images: {tampered_count}")
    
    # Write detailed results to file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write("Image Forgery Detection Results\n")
        f.write("===========================\n\n")
        f.write(f"Total images processed: {len(results)}\n")
        f.write(f"Authentic images: {authentic_count}\n")
        f.write(f"Tampered images: {tampered_count}\n\n")
        f.write("Detailed Results:\n")
        f.write("---------------\n\n")
        
        for image_file, result in results:
            label = "Authentic" if result['prediction'] == 0 else "Tampered"
            confidence = result['prob_authentic'] if result['prediction'] == 0 else result['prob_tampered']
            f.write(f"Image: {image_file}\n")
            f.write(f"Prediction: {label}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
            f.write("\n")
    
    print(f"Detailed results saved to {os.path.join(output_dir, 'results.txt')}")

def main():
    parser = argparse.ArgumentParser(description='Image Forgery Detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image_dir', type=str, help='Path to directory containing images')
    parser.add_argument('--model', type=str, default='forgery_detection_model.pth', help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    if args.image:
        # Process single image
        process_image(args.image, args.model, args.output_dir, args.device)
    elif args.image_dir:
        # Process all images in directory
        batch_process(args.image_dir, args.model, args.output_dir, args.device)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()